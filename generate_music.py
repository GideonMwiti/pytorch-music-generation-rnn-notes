#!/usr/bin/env python3
"""
Interactive CLI for Music Generation with MusicRNN
==================================================
Generates novel musical note sequences, exports standard MIDI (.mid) files,
and renders acoustic piano audio (.wav).

Usage:
    # Free-form generation
    python generate_music.py --length 48 --temperature 0.75 --output-midi outputs/my_tune.mid --output-wav outputs/my_tune.wav

    # Conditioned with custom melody primer (Middle C -> E -> G -> C)
    python generate_music.py --primer "60,64,67,72" --temperature 0.7
"""

import os
import sys
import argparse
import torch

from main import (
    MusicRNN,
    build_vocabulary,
    generate_music_sequence,
    export_to_midi,
    export_to_audio_wav,
    parse_token,
    START_TOKEN,
    PITCH_MIN,
    PITCH_MAX
)

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def pitch_to_note_name(pitch: int) -> str:
    """Converts MIDI pitch (e.g. 60) to pitch name (e.g. C4)."""
    octave = (pitch // 12) - 1
    name = NOTE_NAMES[pitch % 12]
    return f"{name}{octave}"


def duration_to_name(dur_16ths: int) -> str:
    """Converts 16th-note count into musical duration name."""
    mapping = {1: '16th', 2: 'Eighth', 4: 'Quarter', 8: 'Half', 16: 'Whole'}
    return mapping.get(dur_16ths, f"{dur_16ths}/16")


def parse_user_primer(primer_str: str):
    """Parses user primer string into valid tokens."""
    tokens = []
    if not primer_str:
        return [START_TOKEN]

    items = primer_str.split(',')
    for it in items:
        it = it.strip()
        if it.isdigit():
            p = int(it)
            tokens.append(f"N_{p}_4")
        elif it.startswith('N_') or it.startswith('R_'):
            tokens.append(it)
        else:
            # Try to parse e.g. C4, D4
            clean_it = it.upper()
            octave = 4
            if clean_it[-1].isdigit():
                octave = int(clean_it[-1])
                name = clean_it[:-1]
            else:
                name = clean_it
            if name in NOTE_NAMES:
                pitch = (octave + 1) * 12 + NOTE_NAMES.index(name)
                tokens.append(f"N_{pitch}_4")

    if not tokens:
        tokens = [START_TOKEN]
    return tokens


def main():
    parser = argparse.ArgumentParser(description="Interactive MusicRNN Music Generator")
    parser.add_argument('--model', type=str, default='outputs/music_rnn_best.pth', help='Path to trained model checkpoint')
    parser.add_argument('--primer', type=str, default='', help='Comma-separated primer pitches (e.g. "60,64,67,72" or "C4,E4,G4,C5")')
    parser.add_argument('--length', type=int, default=48, help='Number of note tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.75, help='Sampling temperature (0.2=conservative, 1.2=creative)')
    parser.add_argument('--top-p', type=float, default=0.90, help='Nucleus sampling top-p threshold')
    parser.add_argument('--bpm', type=int, default=120, help='Tempo in Beats Per Minute for audio/MIDI')
    parser.add_argument('--output-midi', type=str, default='outputs/interactive_tune.mid', help='Output MIDI path')
    parser.add_argument('--output-wav', type=str, default='outputs/interactive_tune.wav', help='Output WAV audio path')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[!] Model checkpoint not found: {args.model}. Please run main.py first.")
        sys.exit(1)

    device = torch.device(args.device)
    ckpt = torch.load(args.model, map_location=device)

    vocab_tokens = ckpt['vocab_tokens']
    token_to_id = ckpt['token_to_id']
    id_to_token = ckpt['id_to_token']

    model = MusicRNN(
        vocab_size=len(vocab_tokens),
        embedding_dim=128,
        hidden_dim=ckpt['args']['hidden_dim'],
        num_layers=ckpt['args']['num_layers']
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    primer_tokens = parse_user_primer(args.primer)

    print("\n" + "=" * 62)
    print("      MUSICRNN INTERACTIVE MELODIC GENERATOR")
    print("=" * 62)
    print(f"Model Checkpoint : {args.model}")
    print(f"Sampling Temp    : {args.temperature:.2f} | Top-P: {args.top_p:.2f}")
    print(f"Tempo (BPM)      : {args.bpm} BPM")
    print(f"Primer Prompt    : {', '.join(primer_tokens)}")
    print("-" * 62)

    generated = generate_music_sequence(
        model, token_to_id, id_to_token,
        primer=primer_tokens,
        max_tokens=args.length,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device
    )

    # Filter non-note tokens for display
    note_events = []
    for tok in generated:
        is_rest, pitch, dur = parse_token(tok)
        if is_rest:
            note_events.append(('Rest', '-', duration_to_name(dur)))
        elif pitch > 0:
            note_events.append((pitch_to_note_name(pitch), f"MIDI {pitch}", duration_to_name(dur)))

    print(f"\nGenerated Musical Sequence ({len(note_events)} events):")
    print(f"{'Step':<6}{'Note':<10}{'MIDI Pitch':<14}{'Duration':<12}")
    print("-" * 42)
    for idx, (name, pitch_str, dur_name) in enumerate(note_events, 1):
        print(f"#{idx:<5}{name:<10}{pitch_str:<14}{dur_name:<12}")

    # Export to MIDI and WAV
    export_to_midi(generated, args.output_midi)
    export_to_audio_wav(generated, args.output_wav, bpm=args.bpm)

    print("-" * 62)
    print(f"[✓] Standard MIDI File saved to : {args.output_midi}")
    print(f"[✓] Acoustic Piano WAV saved to : {args.output_wav}")
    print("=" * 62 + "\n")


if __name__ == '__main__':
    main()

