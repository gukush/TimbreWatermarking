#!/usr/bin/env python3
import os
import torch
import yaml
import logging
import argparse
import warnings
import numpy as np
#from rich.progress import track
import soundfile
import random
from pathlib import Path
import sys
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "TimbreWatermarking","watermarking_model")))
# Import from TimbreWatermarking repository
from model.conv2_mel_modules import Encoder, Decoder, Discriminator


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    return logging.getLogger(__name__)

def set_seeds(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def load_configs(process_config_path="config/process.yaml",
                model_config_path="config/model.yaml",
                train_config_path="config/train.yaml"):
    """Load and return configuration files."""
    with open(process_config_path) as f:
        process_config = yaml.safe_load(f)
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)
    with open(train_config_path) as f:
        train_config = yaml.safe_load(f)
    return process_config, model_config, train_config

def setup_model(model_config, process_config, train_config, device):
    """Initialize and return the encoder and decoder models."""
    msg_length = train_config["watermark"]["length"]
    win_dim = process_config["audio"]["win_len"]
    embedding_dim = model_config["dim"]["embedding"]
    nlayers_encoder = model_config["layer"]["nlayers_encoder"]
    nlayers_decoder = model_config["layer"]["nlayers_decoder"]
    attention_heads_encoder = model_config["layer"]["attention_heads_encoder"]
    attention_heads_decoder = model_config["layer"]["attention_heads_decoder"]

    encoder = Encoder(process_config, model_config, msg_length, win_dim, embedding_dim,
                     nlayers_encoder=nlayers_encoder,
                     attention_heads=attention_heads_encoder).to(device)
    decoder = Decoder(process_config, model_config, msg_length, win_dim, embedding_dim,
                     nlayers_decoder=nlayers_decoder,
                     attention_heads=attention_heads_decoder).to(device)

    return encoder, decoder

def load_checkpoint(encoder, decoder, model_path,device):
    """Load model checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    model = torch.load(model_path,map_location=device)
    encoder.load_state_dict(model["encoder"])
    decoder.load_state_dict(model["decoder"], strict=False)
    return encoder, decoder

def process_audio(input_dir, output_dir, encoder, device, logger):
    """Process audio files and apply watermark."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get watermark from pool
    #with open("results/wmpool.txt", 'r') as f:
    #    wm = eval(f.readline().strip())  # Using first watermark for simplicity
    wm = eval("[1, 1, 1, 1, 0, 0, 1, 1, 1, 0]")
    msg = torch.from_numpy(np.array([[wm]])).float() * 2 - 1
    msg = msg.to(device)

    # Process each audio file
    # if size > 2116891 then wywali sie na laptopie della i trzeba podzielić
    audio_files = list(input_dir.glob("*.wav"))
    for audio_path in audio_files:#track(audio_files, description="Processing audio files"):
        try:
            print(audio_path)
            output_filename = f"{audio_path.stem}_timbre.wav"
            output_path = output_dir / output_filename
            if output_path.is_file():
                continue
            # Load and process audio
            audio, sr = soundfile.read(str(audio_path))
            print(f"Original audio shape: {audio.shape}")
            max_samples = 2116891
            if audio.shape[0] > max_samples:
                parts = []
                num_samples = audio.shape[0]
                for start in range(0, num_samples, max_samples):
                    end = min(start + max_samples, num_samples)
                    parts.append(audio[start:end,:])
                processed_parts = []
                for part in parts:
                    print(f"Processing part of size: {part.shape}")
                    part_tensor = torch.FloatTensor(part).unsqueeze(0).unsqueeze(0).to(device)
                    with torch.no_grad():
                        part_encoded, carrier_watermarked = encoder.test_forward(part_tensor,msg)
                    processed_parts.append(part_encoded)
                encoded = torch.cat(processed_parts,dim=2)
            else:
                audio_tensor = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0).to(device)

                # Apply watermark
                with torch.no_grad():
                    encoded, carrier_watermarked = encoder.test_forward(audio_tensor, msg)
            # Save watermarked audio


            soundfile.write(str(output_path),
                          encoded.cpu().squeeze(0).squeeze(0).numpy(),
                          samplerate=sr)
            #decoded = decoder.test_forward(encoded)
            logger.info(f"Processed: {audio_path.name}")

        except Exception as e:
            logger.error(f"Error processing {audio_path.name}: {traceback.format_exc()}")

def main():
    parser = argparse.ArgumentParser(description="TimbreWatermark Audio Processor")
    parser.add_argument("--in", dest="input_dir", required=True,
                      help="Input directory containing audio samples")
    parser.add_argument("--out", dest="output_dir", required=True,
                      help="Output directory for transformed audio")
    parser.add_argument("--model", default="results/ckpt/pth/compressed_none-conv2_ep_20_2023-01-17_23_01_01.pth.tar",
                      help="Path to model checkpoint")
    args = parser.parse_args()

    # Setup
    logger = setup_logging()
    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configurations
    configs = load_configs()
    process_config, model_config, train_config = configs

    # Initialize and load model
    encoder, decoder = setup_model(model_config, process_config, train_config, device)
    encoder, decoder = load_checkpoint(encoder, decoder, args.model,device)

    encoder.eval()
    decoder.eval()

    # Process audio files
    process_audio(args.input_dir, args.output_dir, encoder, device, logger)

if __name__ == "__main__":
    main()