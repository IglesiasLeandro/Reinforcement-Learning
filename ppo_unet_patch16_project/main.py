from src import train
import torch
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train.train(device=device)