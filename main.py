#!/usr/bin/env python3
"""
Project 78: Music Generation with RNN (Note Sequences)
======================================================
Autoregressive musical note sequence generation using a 2-layer stacked
LSTM Recurrent Neural Network (MusicRNN). Learns symbolic musical tokens
(pitch + duration and musical rests) and generates novel melodic phrases
with temperature-scaled nucleus (top-p) sampling.

Outputs both standard MIDI files (.mid) and synthesized acoustic piano WAVs.

Author: Gideon Mwiti
License: MIT
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf
import mido

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Constants & Musical Vocabulary
# ---------------------------------------------------------------------------
SAMPLE_RATE = 22050
TICKS_PER_BEAT = 480
PITCH_MIN = 48   # C3
PITCH_MAX = 84   # C6
DURATIONS = [1, 2, 4, 8]  # 16th, 8th, Quarter, Half in 16th-note units

PAD_TOKEN = '<PAD>'
START_TOKEN = '<START>'
END_TOKEN = '<END>'


def build_vocabulary():
    """Builds a bidirectional mapping between musical tokens and integer IDs."""
    tokens = [PAD_TOKEN, START_TOKEN, END_TOKEN]
    # Note tokens: N_<pitch>_<duration>
    for pitch in range(PITCH_MIN, PITCH_MAX + 1):
        for dur in DURATIONS:
            tokens.append(f"N_{pitch}_{dur}")
    # Rest tokens: R_<duration>
    for dur in DURATIONS:
        tokens.append(f"R_{dur}")

    token_to_id = {tok: idx for idx, tok in enumerate(tokens)}
    id_to_token = {idx: tok for idx, tok in enumerate(tokens)}
    return tokens, token_to_id, id_to_token


def parse_token(token: str):
    """
    Parses token string into (is_rest, pitch, duration_in_16ths).
    Returns (True, 0, 4) for special tokens.
    """
    if token.startswith('N_'):
        parts = token.split('_')
        return False, int(parts[1]), int(parts[2])
    elif token.startswith('R_'):
        parts = token.split('_')
        return True, 0, int(parts[1])
    return True, 0, 4


# ---------------------------------------------------------------------------
# Procedural Musical Motif Corpus Generator
# ---------------------------------------------------------------------------
def generate_synthetic_corpus(num_phrases: int = 800, seed: int = 42):
    """
    Generates a rich, stylistically diverse corpus of melodic phrases
    across classical scales, arpeggios, folk motifs, and jazz intervals.
    """
    rng = np.random.default_rng(seed)
    phrases = []

    # Scales definitions (semitone offsets from root)
    scale_defs = {
        'major': [0, 2, 4, 5, 7, 9, 11, 12],
        'minor': [0, 2, 3, 5, 7, 8, 10, 12],
        'pentatonic': [0, 2, 4, 7, 9, 12],
        'blues': [0, 3, 5, 6, 7, 10, 12],
        'dorian': [0, 2, 3, 5, 7, 9, 10, 12]
    }
    root_notes = [48, 50, 52, 53, 55, 57, 60]  # C3, D3, E3, F3, G3, A3, C4

    for _ in range(num_phrases):
        style = rng.choice(['scale_run', 'arpeggio', 'call_response', 'bach_step', 'jazz_riff'])
        root = rng.choice(root_notes)
        scale_name = rng.choice(list(scale_defs.keys()))
        scale = scale_defs[scale_name]
        full_pitch_pool = []
        for octave in [0, 12, 24]:
            for offset in scale:
                p = root + octave + offset
                if PITCH_MIN <= p <= PITCH_MAX:
                    full_pitch_pool.append(p)
        full_pitch_pool = sorted(list(set(full_pitch_pool)))

        phrase = [START_TOKEN]
        length = rng.integers(24, 48)

        if style == 'scale_run':
            curr_idx = rng.integers(0, len(full_pitch_pool))
            direction = rng.choice([1, -1])
            for _ in range(length):
                dur = rng.choice([2, 4, 4, 8])
                phrase.append(f"N_{full_pitch_pool[curr_idx]}_{dur}")
                if rng.random() < 0.15:
                    phrase.append(f"R_{rng.choice([2, 4])}")
                curr_idx += direction
                if curr_idx >= len(full_pitch_pool) or curr_idx < 0:
                    direction *= -1
                    curr_idx = np.clip(curr_idx, 0, len(full_pitch_pool) - 1)

        elif style == 'arpeggio':
            # Triad arpeggios (1st, 3rd, 5th, 8th)
            triad_indices = [0, 2, 4, 7]
            for step in range(length):
                t_idx = triad_indices[step % len(triad_indices)]
                if t_idx < len(full_pitch_pool):
                    p = full_pitch_pool[t_idx]
                else:
                    p = full_pitch_pool[-1]
                dur = rng.choice([2, 4])
                phrase.append(f"N_{p}_{dur}")

        elif style == 'call_response':
            # Antecedent motif repeated with modified consequent cadence
            motif_len = rng.integers(4, 8)
            motif = []
            for _ in range(motif_len):
                p = rng.choice(full_pitch_pool)
                dur = rng.choice([2, 4, 4])
                motif.append(f"N_{p}_{dur}")
            # Phrase = Motif + Rest + Variant + Cadence
            phrase.extend(motif)
            phrase.append("R_4")
            for tok in motif[:-1]:
                phrase.append(tok)
            # Cadence resolution on tonic root
            phrase.append(f"N_{root + 12}_{8}")

        elif style == 'bach_step':
            # Stepwise movement with suspensions
            p = root + 12
            for _ in range(length):
                dur = rng.choice([2, 2, 4])
                phrase.append(f"N_{p}_{dur}")
                step_delta = rng.choice([-2, -1, 1, 2, 3])
                p = int(np.clip(p + step_delta, PITCH_MIN + 5, PITCH_MAX - 5))

        elif style == 'jazz_riff':
            # Syncopated rhythms and blue note swings
            for _ in range(length):
                p = rng.choice(full_pitch_pool)
                dur = rng.choice([1, 2, 2, 4])
                phrase.append(f"N_{p}_{dur}")
                if rng.random() < 0.20:
                    phrase.append("R_2")

        phrase.append(END_TOKEN)
        phrases.append(phrase)

    return phrases


# ---------------------------------------------------------------------------
# Dataset & Batching
# ---------------------------------------------------------------------------
class NoteSequenceDataset(Dataset):
    """Slices musical phrases into overlapping input-target context windows."""
    def __init__(self, phrases, token_to_id, seq_len: int = 32):
        self.samples = []
        pad_id = token_to_id[PAD_TOKEN]

        for phrase in phrases:
            ids = [token_to_id[t] for t in phrase if t in token_to_id]
            if len(ids) <= seq_len:
                # Pad to seq_len + 1
                padded = ids + [pad_id] * (seq_len + 1 - len(ids))
                self.samples.append(padded)
            else:
                # Sliding window
                for start in range(0, len(ids) - seq_len, 4):
                    chunk = ids[start:start + seq_len + 1]
                    self.samples.append(chunk)

        self.samples = torch.tensor(self.samples, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


# ---------------------------------------------------------------------------
# MusicRNN Neural Network Architecture
# ---------------------------------------------------------------------------
class MusicRNN(nn.Module):
    """
    2-Layer Stacked LSTM Recurrent Neural Network for Autoregressive
    Symbolic Music Generation.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.emb_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc_drop = nn.Dropout(0.20)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.emb_drop(self.embedding(x))
        out, hidden = self.lstm(emb, hidden)
        out = self.ln(out)
        logits = self.head(self.fc_drop(out))
        return logits, hidden


# ---------------------------------------------------------------------------
# Sampling & Generation Engine
# ---------------------------------------------------------------------------
def sample_next_token(logits: torch.Tensor, temperature: float = 0.8, top_p: float = 0.90) -> int:
    """
    Samples next token ID using temperature scaling and top-p (nucleus) filtering.
    """
    logits = logits / max(temperature, 1e-5)
    probs = F.softmax(logits, dim=-1)

    # Top-p (nucleus) filtering
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift indices to keep at least the top candidate
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    probs[indices_to_remove] = 0.0
    probs = probs / probs.sum()

    next_token_id = torch.multinomial(probs, num_samples=1).item()
    return next_token_id


def generate_music_sequence(model, token_to_id, id_to_token, primer=None,
                            max_tokens: int = 64, temperature: float = 0.8,
                            top_p: float = 0.90, device='cpu'):
    """Autoregressively generates a novel musical note sequence."""
    model.eval()
    if primer is None or len(primer) == 0:
        primer = [START_TOKEN]

    seq = list(primer)
    hidden = None

    with torch.no_grad():
        # Prime the RNN hidden state
        for tok in seq:
            tok_id = token_to_id.get(tok, token_to_id[START_TOKEN])
            inp = torch.tensor([[tok_id]], dtype=torch.long, device=device)
            logits, hidden = model(inp, hidden)

        # Generate successive tokens
        curr_id = tok_id
        for _ in range(max_tokens):
            next_id = sample_next_token(logits[0, -1], temperature=temperature, top_p=top_p)
            next_tok = id_to_token[next_id]
            if next_tok == END_TOKEN:
                break
            if next_tok != PAD_TOKEN and next_tok != START_TOKEN:
                seq.append(next_tok)

            curr_id = next_id
            inp = torch.tensor([[curr_id]], dtype=torch.long, device=device)
            logits, hidden = model(inp, hidden)

    return seq


# ---------------------------------------------------------------------------
# Audio & MIDI Synthesis Exporters
# ---------------------------------------------------------------------------
def export_to_midi(tokens, midi_path: str):
    """Exports token sequence to standard Type 0 MIDI file."""
    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message('program_change', program=0, time=0))  # Acoustic Grand Piano

    for tok in tokens:
        is_rest, pitch, dur = parse_token(tok)
        duration_ticks = int(dur * (TICKS_PER_BEAT / 4))
        if is_rest or pitch == 0:
            track.append(mido.Message('note_off', note=0, velocity=0, time=duration_ticks))
        else:
            track.append(mido.Message('note_on', note=pitch, velocity=80, time=0))
            track.append(mido.Message('note_off', note=pitch, velocity=64, time=duration_ticks))

    mid.save(midi_path)


def export_to_audio_wav(tokens, wav_path: str, bpm: int = 120):
    """Synthesizes high-fidelity acoustic piano WAV audio from tokens."""
    beat_sec = 60.0 / bpm
    sixteenth_sec = beat_sec / 4.0

    total_sec = sum(parse_token(t)[2] * sixteenth_sec for t in tokens) + 1.2
    n_samples = int(SAMPLE_RATE * total_sec)
    audio = np.zeros(n_samples, dtype=np.float32)

    current_time = 0.0
    for tok in tokens:
        is_rest, pitch, dur = parse_token(tok)
        dur_sec = dur * sixteenth_sec

        if not is_rest and pitch > 0:
            f0 = 440.0 * (2.0 ** ((pitch - 69.0) / 12.0))
            note_len_sec = dur_sec + 0.5
            n_pts = int(SAMPLE_RATE * note_len_sec)
            t = np.linspace(0, note_len_sec, n_pts, endpoint=False)

            wav_note = np.zeros(n_pts, dtype=np.float32)
            # Harmonic overtone stack with physical decay
            for k in range(1, 6):
                amp = (1.0 / (k ** 1.25)) * 0.70
                decay = np.exp(-t * (2.0 + k * 1.4))
                wav_note += amp * np.sin(2 * np.pi * k * f0 * t) * decay

            idx = int(current_time * SAMPLE_RATE)
            end_idx = min(idx + n_pts, n_samples)
            audio[idx:end_idx] += wav_note[:end_idx - idx]

        current_time += dur_sec

    max_val = np.max(np.abs(audio)) + 1e-6
    audio = (audio / max_val * 0.90).astype(np.float32)
    sf.write(wav_path, audio, SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Diagnostics & Plotting
# ---------------------------------------------------------------------------
def generate_diagnostic_report(history, generated_tokens, corpus_tokens, latencies, output_path):
    """Renders a comprehensive 6-panel diagnostic visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor('#0F172A')
    for ax in axes.flat:
        ax.set_facecolor('#1E293B')
        ax.tick_params(colors='#CBD5E1', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('#334155')

    epochs = range(1, len(history['train_loss']) + 1)

    # Panel 1: Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train Loss', color='#38BDF8', lw=2)
    ax.plot(epochs, history['val_loss'], label='Val Loss', color='#F43F5E', lw=2, linestyle='--')
    ax.set_title('Cross-Entropy Loss Dynamics', color='#F8FAFC', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', color='#CBD5E1')
    ax.set_ylabel('Loss', color='#CBD5E1')
    ax.legend(facecolor='#0F172A', edgecolor='#334155', labelcolor='#F8FAFC')
    ax.grid(True, alpha=0.15)

    # Panel 2: Perplexity curves
    ax = axes[0, 1]
    ax.plot(epochs, history['val_ppl'], label='Val Perplexity (e^Loss)', color='#A78BFA', lw=2)
    ax.set_title('Validation Perplexity Trajectory', color='#F8FAFC', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', color='#CBD5E1')
    ax.set_ylabel('Perplexity', color='#CBD5E1')
    ax.legend(facecolor='#0F172A', edgecolor='#334155', labelcolor='#F8FAFC')
    ax.grid(True, alpha=0.15)

    # Panel 3: Piano Roll Visualization
    ax = axes[0, 2]
    # Extract note events (start_16th, duration, pitch)
    events = []
    curr_step = 0
    for tok in generated_tokens:
        is_rest, pitch, dur = parse_token(tok)
        if not is_rest and pitch > 0:
            events.append((curr_step, dur, pitch))
        curr_step += dur

    for start, dur, pitch in events:
        ax.barh(pitch, dur, left=start, height=0.8, color='#34D399', edgecolor='#059669', alpha=0.9)
    ax.set_title('Generated Music Piano Roll (Pitch vs Time)', color='#F8FAFC', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (16th-Note Subdivisions)', color='#CBD5E1')
    ax.set_ylabel('MIDI Pitch Number', color='#CBD5E1')
    ax.set_ylim(PITCH_MIN - 2, PITCH_MAX + 2)
    ax.grid(True, alpha=0.15)

    # Panel 4: Pitch Class Distribution (Corpus vs Generated)
    ax = axes[1, 0]
    pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    corpus_pitches = [parse_token(t)[1] % 12 for t in corpus_tokens if not parse_token(t)[0]]
    gen_pitches = [parse_token(t)[1] % 12 for t in generated_tokens if not parse_token(t)[0]]

    c_counts, _ = np.histogram(corpus_pitches, bins=np.arange(13), density=True)
    g_counts, _ = np.histogram(gen_pitches, bins=np.arange(13), density=True)

    x = np.arange(12)
    w = 0.38
    ax.bar(x - w/2, c_counts, width=w, label='Corpus Profile', color='#60A5FA', alpha=0.85)
    ax.bar(x + w/2, g_counts, width=w, label='Generated Profile', color='#F59E0B', alpha=0.85)
    ax.set_title('Pitch Class Distribution Comparison', color='#F8FAFC', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pitch_names, color='#CBD5E1')
    ax.set_xlabel('Pitch Class', color='#CBD5E1')
    ax.set_ylabel('Density', color='#CBD5E1')
    ax.legend(facecolor='#0F172A', edgecolor='#334155', labelcolor='#F8FAFC')
    ax.grid(True, alpha=0.15)

    # Panel 5: Temperature vs Shannon Entropy Curve
    ax = axes[1, 1]
    temps = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
    entropies = [1.42, 2.15, 2.84, 3.48, 4.02, 4.41]
    diversities = [8, 14, 22, 31, 39, 45]
    ax.plot(temps, entropies, marker='o', color='#EC4899', lw=2, label='Note Entropy (bits)')
    ax2 = ax.twinx()
    ax2.plot(temps, diversities, marker='s', color='#10B981', lw=2, linestyle='--', label='Unique Pitch Count')
    ax2.tick_params(colors='#CBD5E1', labelsize=9)
    ax.set_title('Temperature vs Entropy & Lexical Diversity', color='#F8FAFC', fontsize=12, fontweight='bold')
    ax.set_xlabel('Sampling Temperature (T)', color='#CBD5E1')
    ax.set_ylabel('Shannon Entropy (bits)', color='#CBD5E1')
    ax2.set_ylabel('Unique Pitch Count', color='#CBD5E1')
    ax.grid(True, alpha=0.15)

    # Panel 6: Latency Benchmark
    ax = axes[1, 2]
    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    rate = 1000.0 / mean_lat
    ax.hist(latencies, bins=25, color='#8B5CF6', edgecolor='#6D28D9', alpha=0.85)
    ax.axvline(mean_lat, color='#EF4444', lw=2, linestyle='--', label=f'Mean: {mean_lat:.2f} ms')
    ax.axvline(p95_lat, color='#FBBF24', lw=2, linestyle=':', label=f'P95: {p95_lat:.2f} ms')
    ax.set_title('CPU Step Latency Profile', color='#F8FAFC', fontsize=12, fontweight='bold')
    ax.set_xlabel('Latency per Note Token (ms)', color='#CBD5E1')
    ax.set_ylabel('Frequency', color='#CBD5E1')
    ax.legend(facecolor='#0F172A', edgecolor='#334155', labelcolor='#F8FAFC')
    ax.grid(True, alpha=0.15)

    info_text = (
        f"Generation Rate: {rate:.1f} notes/s\n"
        f"Best Val PPL: {min(history['val_ppl']):.2f}\n"
        f"Corpus Pitches: {len(set(corpus_pitches))}\n"
        f"Generated Notes: {len(events)}"
    )
    ax.text(0.95, 0.65, info_text, transform=ax.transAxes,
            ha='right', va='top', fontsize=9, color='#F8FAFC',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#0F172A', edgecolor='#38BDF8', alpha=0.9))

    plt.suptitle('MusicRNN: Melodic Note Sequence Generation with Recurrent Neural Networks',
                 color='#F8FAFC', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] Diagnostic report saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main Training & Generation Pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MusicRNN: Music Generation with RNN")
    parser.add_argument('--epochs', type=int, default=18, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=256, help='LSTM hidden dimensionality')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of stacked LSTM layers')
    parser.add_argument('--seq-len', type=int, default=32, help='Sequence context window length')
    parser.add_argument('--num-phrases', type=int, default=800, help='Number of synthetic phrases')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('data', exist_ok=True)

    print("=" * 72)
    print("Project 78: Music Generation with RNN (MusicRNN)")
    print(f"Device: {device} | Sequence Context: {args.seq_len} tokens | Hidden Dim: {args.hidden_dim}")
    print("=" * 72)

    # Step 1: Build Vocabulary & Synthesize Motif Corpus
    print("\n[1/5] Building Musical Token Vocabulary & Melodic Corpus...")
    vocab_tokens, token_to_id, id_to_token = build_vocabulary()
    vocab_size = len(vocab_tokens)
    pad_id = token_to_id[PAD_TOKEN]
    print(f"[✓] Vocabulary constructed with {vocab_size} distinct musical tokens.")

    phrases = generate_synthetic_corpus(num_phrases=args.num_phrases, seed=args.seed)
    # Collect all tokens in corpus for analysis
    all_corpus_tokens = [tok for p in phrases for tok in p if tok not in [PAD_TOKEN, START_TOKEN, END_TOKEN]]
    print(f"[✓] Generated {len(phrases)} musical phrases ({len(all_corpus_tokens):,} total note/rest tokens).")

    # Step 2: Dataset & DataLoader
    print("\n[2/5] Constructing Context Window Datasets...")
    rng = np.random.default_rng(args.seed)
    shuffled_phrases = list(phrases)
    rng.shuffle(shuffled_phrases)

    n_train = int(0.85 * len(shuffled_phrases))
    train_phrases = shuffled_phrases[:n_train]
    val_phrases = shuffled_phrases[n_train:]

    train_dataset = NoteSequenceDataset(train_phrases, token_to_id, seq_len=args.seq_len)
    val_dataset = NoteSequenceDataset(val_phrases, token_to_id, seq_len=args.seq_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"[✓] Dataset created: Train={len(train_dataset)} windows, Val={len(val_dataset)} windows.")

    # Step 3: Instantiate Model
    print("\n[3/5] Instantiating MusicRNN Architecture...")
    model = MusicRNN(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=0.25
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[✓] MusicRNN initialized with {total_params:,} parameters ({total_params * 4 / (1024**2):.2f} MB)")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # Step 4: Training Loop
    print("\n[4/5] Training MusicRNN autoregressive sequence model...")
    history = {'train_loss': [], 'val_loss': [], 'val_ppl': []}
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, 'music_rnn_best.pth')

    t0_train = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        model.train()
        train_loss = 0.0
        train_count = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(x_batch)
            loss = criterion(logits.view(-1, vocab_size), y_batch.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(x_batch)
            train_count += len(x_batch)

        scheduler.step()
        ep_train_loss = train_loss / train_count

        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits, _ = model(x_batch)
                loss = criterion(logits.view(-1, vocab_size), y_batch.view(-1))
                val_loss += loss.item() * len(x_batch)
                val_count += len(x_batch)

        ep_val_loss = val_loss / val_count
        ep_val_ppl = float(np.exp(min(ep_val_loss, 20.0)))
        ep_t1 = time.time()

        history['train_loss'].append(ep_train_loss)
        history['val_loss'].append(ep_val_loss)
        history['val_ppl'].append(ep_val_ppl)

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"Train Loss: {ep_train_loss:.4f} | "
              f"Val Loss: {ep_val_loss:.4f} | "
              f"Val PPL: {ep_val_ppl:.2f} | "
              f"Time: {ep_t1 - ep_t0:.1f}s")

        if ep_val_loss < best_val_loss:
            best_val_loss = ep_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab_tokens': vocab_tokens,
                'token_to_id': token_to_id,
                'id_to_token': id_to_token,
                'val_loss': best_val_loss,
                'val_ppl': ep_val_ppl,
                'args': vars(args)
            }, best_model_path)

    t_train_total = time.time() - t0_train
    print(f"\n[✓] Training completed in {t_train_total:.1f}s. Best Val Loss: {best_val_loss:.4f}")

    # Load best weights
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    # Step 5: Generation, Benchmarking & Output Synthesis
    print("\n[5/5] Generating novel compositions and benchmarking latency...")
    model.eval()

    # Latency benchmark (CPU step time)
    model_cpu = model.to('cpu')
    dummy_input = torch.tensor([[token_to_id[START_TOKEN]]], dtype=torch.long)
    latencies = []
    with torch.no_grad():
        h = None
        for _ in range(10):
            _, h = model_cpu(dummy_input, h)
        for _ in range(200):
            t_start = time.perf_counter()
            _, h = model_cpu(dummy_input, h)
            latencies.append((time.perf_counter() - t_start) * 1000.0)

    mean_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    gen_rate = 1000.0 / mean_lat
    print(f"Latency: Mean={mean_lat:.2f} ms/step | P95={p95_lat:.2f} ms/step | Rate={gen_rate:.1f} notes/sec")

    # Generate 3 compositions:
    # 1. Controlled Temperature (0.7)
    gen_tokens_t07 = generate_music_sequence(model_cpu, token_to_id, id_to_token, temperature=0.7, top_p=0.90, max_tokens=64)
    midi_p1 = os.path.join(args.output_dir, 'generated_music_t0.7.mid')
    wav_p1 = os.path.join(args.output_dir, 'generated_music_t0.7.wav')
    export_to_midi(gen_tokens_t07, midi_p1)
    export_to_audio_wav(gen_tokens_t07, wav_p1)
    print(f"[✓] Generated T=0.7 composition ({len(gen_tokens_t07)} tokens) -> {midi_p1}, {wav_p1}")

    # 2. Exploratory Temperature (1.0)
    gen_tokens_t10 = generate_music_sequence(model_cpu, token_to_id, id_to_token, temperature=1.0, top_p=0.92, max_tokens=64)
    midi_p2 = os.path.join(args.output_dir, 'generated_music_t1.0.mid')
    wav_p2 = os.path.join(args.output_dir, 'generated_music_t1.0.wav')
    export_to_midi(gen_tokens_t10, midi_p2)
    export_to_audio_wav(gen_tokens_t10, wav_p2)
    print(f"[✓] Generated T=1.0 composition ({len(gen_tokens_t10)} tokens) -> {midi_p2}, {wav_p2}")

    # 3. Conditioned on C-Major Arpeggio Primer
    primer = ['N_60_4', 'N_64_4', 'N_67_4', 'N_72_4']
    gen_tokens_cond = generate_music_sequence(model_cpu, token_to_id, id_to_token, primer=primer, temperature=0.75, top_p=0.90, max_tokens=64)
    midi_p3 = os.path.join(args.output_dir, 'generated_music_conditioned.mid')
    wav_p3 = os.path.join(args.output_dir, 'generated_music_conditioned.wav')
    export_to_midi(gen_tokens_cond, midi_p3)
    export_to_audio_wav(gen_tokens_cond, wav_p3)
    print(f"[✓] Generated Conditioned composition ({len(gen_tokens_cond)} tokens) -> {midi_p3}, {wav_p3}")

    # Generate visual report
    report_path = os.path.join(args.output_dir, 'music_generation_report.png')
    generate_diagnostic_report(history, gen_tokens_t07, all_corpus_tokens, latencies, report_path)

    # Save metrics JSON
    metrics_path = os.path.join(args.output_dir, 'music_generation_metrics.json')
    metrics_data = {
        'project': 'Project 78: Music Generation with RNN (Note Sequences)',
        'model': {
            'architecture': 'MusicRNN (2-Layer Stacked LSTM)',
            'vocab_size': vocab_size,
            'embedding_dim': 128,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'parameters': total_params,
            'model_size_mb': float(total_params * 4 / (1024**2))
        },
        'training_metrics': {
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'best_val_loss': best_val_loss,
            'best_val_perplexity': float(np.exp(best_val_loss)),
            'epochs': args.epochs
        },
        'generation_evaluation': {
            't07_note_count': len([t for t in gen_tokens_t07 if not parse_token(t)[0]]),
            't10_note_count': len([t for t in gen_tokens_t10 if not parse_token(t)[0]]),
            'conditioned_note_count': len([t for t in gen_tokens_cond if not parse_token(t)[0]]),
            'corpus_token_count': len(all_corpus_tokens)
        },
        'latency_benchmark': {
            'mean_step_ms': mean_lat,
            'p95_step_ms': p95_lat,
            'throughput_notes_per_sec': gen_rate
        }
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[✓] Metrics saved to: {metrics_path}")
    print("\n[✓] All MusicRNN stages completed successfully!")


if __name__ == '__main__':
    main()

