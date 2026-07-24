import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            LinearBlock(self.input_dim, 2*self.latent_dim),
            LinearBlock(2*self.latent_dim, 2*self.latent_dim),
            nn.Linear(2*self.latent_dim, 2*self.latent_dim)
        )
        self.decoder = nn.Sequential(
            LinearBlock(self.latent_dim, 2*self.latent_dim),
            LinearBlock(2*self.latent_dim, 2*self.latent_dim),
            nn.Linear(2*self.latent_dim, self.input_dim)
        )

    def forward(self, x):
        mu, log2_sigma = torch.split(self.encoder(x), [self.latent_dim, self.latent_dim], dim=1)

        # sample random z
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * log2_sigma)

        # reconstruct
        x_rec = self.decoder(z)
        return x_rec, mu, log2_sigma

class ELBO_Loss(nn.Module):
    def __init__(self):
        super(ELBO_Loss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.sigma_d = 0.7

    def forward(self, x_rec, x_tar, mu, log2_sigma):
        # batch size
        N = x_rec.shape[0]

        rec_loss = self.mse(x_rec, x_tar) / (2 * self.sigma_d**2)
        kl_loss  = - 0.5*(1 + log2_sigma - mu**2 - torch.exp(log2_sigma)).sum()

        return (rec_loss + kl_loss) / N

def train_vae(train_loader, test_loader, model, optimizer, epochs):
    elbo_loss = ELBO_Loss()

    for epoch in range(epochs):
        # train phase
        model.train()

        for batch_idx, (x, _) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.reshape(-1, 28*28).to(device)
            x_rec, mu, log2_sigma = model(x)
            loss = elbo_loss(x_rec, x, mu, log2_sigma)
            loss.backward()
            optimizer.step()

            # print batch statistics
            if batch_idx % 400 == 0:
                print(f'\tepoch: {epoch} batch: {batch_idx}: loss: {loss.item():.4f}')

        # test phase
        model.eval()
        eval_loss = 0
        for batch_idx, (x, _) in enumerate(test_loader):
            with torch.no_grad():
                x = x.reshape(-1, 28*28).to(device)
                x_rec, mu, log2_sigma = model(x)
                loss = elbo_loss(x_rec, x, mu, log2_sigma)
                eval_loss += loss.item()

        eval_loss /= len(test_loader)
        print(f' ==== Epoch {epoch+1}/{epochs} | Loss: {eval_loss:.4f}')

        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'vae.pth')
            with torch.no_grad():
                z = torch.randn(100, model.latent_dim).to(device)
                samples = model.decoder(z).view(-1, 1, 28, 28)
                grid = make_grid(samples, nrow=10, normalize=True, pad_value=1, padding=2)
                grid = torch.nn.functional.interpolate(
                    grid.unsqueeze(0), scale_factor=2, mode='bilinear', align_corners=False
                ).squeeze(0)
                save_image(grid, f'sample_epoch_{epoch:03d}.png')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(28*28,16).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    print('Total model size:', total_params, "encoder:", encoder_params, "decoder:", decoder_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    elbo_loss = ELBO_Loss()

    transform = transforms.ToTensor()

    train_set = datasets.MNIST('../data', train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST('../data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=80, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=80, shuffle=False)

    train_vae(train_loader, test_loader, model, optimizer, 100)