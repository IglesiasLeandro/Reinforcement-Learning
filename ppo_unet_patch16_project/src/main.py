from .train import train

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(device=device)