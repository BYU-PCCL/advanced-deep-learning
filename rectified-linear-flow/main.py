import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import torchvision
import torchvision.transforms as transforms


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        context_dim,
        h,
        out_dim,
    ):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim + context_dim, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, out_dim),
        )

    def forward(self, x, context):
        return self.network(torch.cat((x, context), dim=1))


class ConvBlock(nn.Module):
    """Convolutional block with time embedding."""

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        # Add time embedding
        time_emb = self.time_mlp(t_emb)
        h = h + time_emb[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h


class SimpleUNet(nn.Module):
    """Simplified U-Net for MNIST (28x28 images)."""

    def __init__(self, in_channels=1, time_emb_dim=32):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder
        self.enc1 = ConvBlock(in_channels, 32, time_emb_dim)
        self.enc2 = ConvBlock(32, 64, time_emb_dim)
        self.enc3 = ConvBlock(64, 128, time_emb_dim)

        # Bottleneck
        self.bottleneck = ConvBlock(128, 128, time_emb_dim)

        # Decoder
        self.dec3 = ConvBlock(256, 64, time_emb_dim)
        self.dec2 = ConvBlock(128, 32, time_emb_dim)
        self.dec1 = ConvBlock(64, 32, time_emb_dim)

        # Output
        self.out = nn.Conv2d(32, in_channels, 1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t)

        # Encoder
        e1 = self.enc1(x, t_emb)  # 28x28
        e2 = self.enc2(self.pool(e1), t_emb)  # 14x14
        e3 = self.enc3(self.pool(e2), t_emb)  # 7x7

        # Bottleneck
        b = self.bottleneck(self.pool(e3), t_emb)  # 3x3

        # Decoder with skip connections - use interpolate with exact sizes
        d3 = nn.functional.interpolate(b, size=e3.shape[2:], mode="nearest")
        d3 = self.dec3(torch.cat([d3, e3], dim=1), t_emb)  # 7x7

        d2 = nn.functional.interpolate(d3, size=e2.shape[2:], mode="nearest")
        d2 = self.dec2(torch.cat([d2, e2], dim=1), t_emb)  # 14x14

        d1 = nn.functional.interpolate(d2, size=e1.shape[2:], mode="nearest")
        d1 = self.dec1(torch.cat([d1, e1], dim=1), t_emb)  # 28x28

        return self.out(d1)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dist1, dist2):
        self.dist1 = dist1
        self.dist2 = dist2
        assert self.dist1.shape == self.dist2.shape

    def __len__(self):
        return self.dist1.shape[0]

    def __getitem__(self, idx):
        return self.dist1[idx], self.dist2[idx]


def sample_multimodal_distribution(modes, std, batch_size=1000):
    dataset = []
    for i in range(batch_size):
        sample = np.random.randn(modes.shape[1]) * std
        mode_idx = np.random.randint(modes.shape[0])
        sample[0] += modes[mode_idx, 0]
        sample[1] += modes[mode_idx, 1]
        dataset.append(sample)
    return np.array(dataset, dtype="float32")


# Part 2
def train_rectified_flow(
    rectified_flow,
    optimizer,
    train_dataloader,
    NB_EPOCHS,
    device,
    eps=1e-15,
):
    for epoch in tqdm(
        range(NB_EPOCHS),
        desc=f"Training Reflow Step",
    ):
        for z0, z1 in train_dataloader:
            z0, z1 = (
                z0.to(device),
                z1.to(device),
            )
            # z0 shape: [batch_size, features]
            # z1 shape: [batch_size, features]
            batch_size = z1.shape[0]
            # TODO: Part 2
            ########################
            # Sample a random time t for each sample in the batch [0, 1]
            t = torch.rand(
                (batch_size, 1),
                device=device,
            )
            # Compute the interpolated point z_t at time t between z0 and z1
            z_t = t * z1 + (1.0 - t) * z0

            # Compute the target vector field
            target = z1 - z0

            # Predict the vector field at interpolated points
            pred = rectified_flow(z_t, t)

            # MSE
            loss = (
                (target - pred)
                .view(batch_size, -1)
                .abs()
                .pow(2)
                .sum(dim=1)
                .mean()
            )
            ##########################

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# Part 3
def sample(rectified_flow, T, pi_0_path):
    """
    Args:
        rectified_flow: The trained rectified flow model.
        T: Number of time steps for sampling.
        pi_0_path: Initial points for the path.

    Returns:
        A tensor containing the sampled paths.

    Instructions:

    """
    # pi_0_path shape: [batch_size, features]
    batch_size = pi_0_path.size(0)
    samples = [pi_0_path.clone().unsqueeze(0)]
    # TODO: Part 3
    #####################
    for i in range(T):
        last_sample = samples[-1]
        t = (
            torch.ones(
                (batch_size, 1),
                device=last_sample.device,
            )
            * i
            / T
        )
        current_pos = last_sample.squeeze(0)
        drift_pred = rectified_flow(current_pos, t)
        samples.append((last_sample + drift_pred * 1.0 / T))
    return torch.cat(samples)
    ####################


# Part 4
def train_rectified_flow_mnist(
    rectified_flow,
    optimizer,
    train_dataloader,
    NB_EPOCHS,
    device,
):
    """Training loop for MNIST images."""
    for epoch in tqdm(range(NB_EPOCHS), desc="Training MNIST Rectified Flow"):
        total_loss = 0
        num_batches = 0
        for batch_idx, (z1, _) in enumerate(train_dataloader):
            # TODO: Part 4
            ##############
            z1 = z1.to(device)
            batch_size = z1.shape[0]

            # Sample random noise as z0
            z0 = torch.randn_like(z1)
            # Sample random time
            t = torch.rand((batch_size, 1), device=device)

            # Interpolate
            z_t = t.view(-1, 1, 1, 1) * z1 + (1.0 - t.view(-1, 1, 1, 1)) * z0
            # Target velocity
            target = z1 - z0

            # Predict velocity
            pred = rectified_flow(z_t, t)
            # Loss
            loss = (target - pred).pow(2).mean()
            ##############

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{NB_EPOCHS}, Loss: {avg_loss:.6f}")


# Part 4
def sample_mnist(rectified_flow, T, batch_size, device):
    """Sample MNIST images starting from noise."""
    # Start from random noise
    z = torch.randn(batch_size, 1, 28, 28, device=device)

    for i in tqdm(range(T), desc="Sampling"):
        t = torch.ones((batch_size, 1), device=device) * i / T
        with torch.no_grad():
            drift = rectified_flow(z, t)
            z = z + drift * (1.0 / T)

    return z


def visualize_linear_interpolation(
    pi_0,
    pi_1,
    save_path,
    plot_trajectory_stride,
):
    """Visualize linear interpolation between pi_0 and pi_1."""
    print("Plotting linear interpolation...")
    plt.figure(figsize=(10, 10))

    # Plot initial and final points
    plt.scatter(
        pi_0[:, 0],
        pi_0[:, 1],
        c="purple",
        label="pi_0",
    )
    plt.scatter(
        pi_1[:, 0],
        pi_1[:, 1],
        c="red",
        label="pi_1",
    )

    # Plot trajectories for a subset of points for clarity
    for i in range(
        0,
        pi_0.shape[0],
        plot_trajectory_stride,
    ):
        plt.plot(
            [pi_0[i, 0], pi_1[i, 0]],
            [pi_0[i, 1], pi_1[i, 1]],
            c="gray",
            linewidth=0.5,
        )

    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Linear interpolation plot saved to {save_path}")


def visualize_flow(
    rectified_flow,
    theta,
    std,
    device,
    save_path,
    num_sampling_steps,
    plot_trajectory_stride,
):
    """Visualize the flow and save to the specified path."""
    plt.figure(figsize=(10, 10))

    for idx, theta_ in enumerate([theta[::2], theta[1::2]]):
        modes = np.array(
            [
                (12.0 * x, 12.0 * y)
                for x, y in zip(
                    np.cos(theta_),
                    np.sin(theta_),
                )
            ]
        )

        test_pi_0 = sample_multimodal_distribution(modes, std, batch_size=1000)

        # Generate the flow path using the trained model
        test_pi_1_path = sample(
            rectified_flow,
            num_sampling_steps,
            torch.from_numpy(test_pi_0).to(device),
        )

        # Plot initial points
        plt.scatter(
            test_pi_0[:, 0],
            test_pi_0[:, 1],
            c="purple",
        )
        # Plot final points
        plt.scatter(
            test_pi_1_path[-1, :, 0].data.cpu().numpy(),
            test_pi_1_path[-1, :, 1].data.cpu().numpy(),
            c="red",
        )

        # Plot the trajectories
        print(f"Plotting trajectories for visualization group {idx + 1}...")
        for i in tqdm(range(1, num_sampling_steps, 1)):
            for j in range(
                0,
                test_pi_0.shape[0],
                plot_trajectory_stride,
            ):
                plt.plot(
                    [
                        test_pi_1_path[i - 1, j, 0].item(),
                        test_pi_1_path[i, j, 0].item(),
                    ],
                    [
                        test_pi_1_path[i - 1, j, 1].item(),
                        test_pi_1_path[i, j, 1].item(),
                    ],
                    c="C0" if idx == 0 else "g",
                    linewidth=0.5,
                )

    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")


def visualize_mnist_samples(samples, save_path, nrow=8):
    """Visualize generated MNIST samples."""
    samples = samples.cpu()
    grid = torchvision.utils.make_grid(
        samples, nrow=nrow, normalize=True, value_range=(-1, 1)
    )
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"MNIST samples saved to {save_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ["1", "2", "3", "4"]:
        print("Usage: python main.py <part>")
        print("  part 1: Linear Interpolation")
        print("  part 2: 1-Rectified Flow")
        print("  part 3: 2-Rectified Flow")
        print("  part 4: MNIST Generation")
        sys.exit(1)

    part = sys.argv[1]

    # Create images directory if it doesn't exist
    os.makedirs("./images", exist_ok=True)

    # --- Constants ---
    device = "cuda"
    batch_size = 2048
    dataset_size = 10_000
    nb_epochs = 500
    wd = 0.01
    NUM_SAMPLING_STEPS = 250  # For ODE solver
    PLOT_TRAJECTORY_STRIDE = 10  # For visualization clarity

    # --- Initial Data Distributions ---
    theta = (
        np.array(
            [
                0.0,
                60,
                120,
                180,
                240,
                300,
            ]
        )
        / 360
        * 2
        * np.pi
    )
    std = 0.5
    radius = 12.0
    modes = np.array(
        [
            (radius * x, radius * y)
            for x, y in zip(
                np.cos(theta),
                np.sin(theta),
            )
        ]
    )

    # pi_0 is the initial source distribution, which will remain constant
    pi_0 = sample_multimodal_distribution(
        modes,
        std,
        batch_size=dataset_size,
    )

    radius = 5.0
    modes = np.array(
        [
            (radius * x, radius * y)
            for x, y in zip(
                np.cos(theta),
                np.sin(theta),
            )
        ]
    )
    # pi_1 is our initial target
    pi_1 = sample_multimodal_distribution(
        modes,
        std,
        batch_size=dataset_size,
    )

    # 1. Shuffle pi_1 to ensure random pairing
    np.random.shuffle(pi_1)

    # Convert to PyTorch tensors
    current_pi_0 = torch.from_numpy(pi_0).float()
    current_pi_1 = torch.from_numpy(pi_1).float()

    # --- Part 1: Linear Interpolation ---
    if part == "1":
        print("\n--- Running Part 1: Linear Interpolation ---")
        visualize_linear_interpolation(
            current_pi_0.numpy(),
            current_pi_1.numpy(),
            "./images/flow0_linear_interpolation.png",
            PLOT_TRAJECTORY_STRIDE,
        )

    # --- Part 2: 1-Rectified Flow ---
    elif part == "2":
        print("\n--- Running Part 2: 1-Rectified Flow ---")
        rectified_flow = MLP(2, 1, 64, 2).to(device)
        optimizer = torch.optim.Adam(
            rectified_flow.parameters(),
            lr=5e-3,
            weight_decay=wd,
        )
        dataset = Dataset(current_pi_0, current_pi_1)
        train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        train_rectified_flow(
            rectified_flow,
            optimizer,
            train_dataloader,
            nb_epochs,
            device,
        )
        visualize_flow(
            rectified_flow,
            theta,
            std,
            device,
            "./images/flow1.png",
            NUM_SAMPLING_STEPS,
            PLOT_TRAJECTORY_STRIDE,
        )

    # --- Part 3: 2-Rectified Flow ---
    elif part == "3":
        print("\n--- Running Part 3: 2-Rectified Flow ---")
        REFLOW_STEPS = 2
        for step in range(REFLOW_STEPS):
            print(f"\n--- Starting Reflow Step {step + 1}/{REFLOW_STEPS} ---")
            rectified_flow = MLP(2, 1, 64, 2).to(device)
            optimizer = torch.optim.Adam(
                rectified_flow.parameters(),
                lr=5e-3,
                weight_decay=wd,
            )
            dataset = Dataset(
                current_pi_0,
                current_pi_1,
            )
            train_dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
            )
            train_rectified_flow(
                rectified_flow,
                optimizer,
                train_dataloader,
                nb_epochs,
                device,
            )
            visualize_flow(
                rectified_flow,
                theta,
                std,
                device,
                f"./images/flow{step + 1}.png",
                NUM_SAMPLING_STEPS,
                PLOT_TRAJECTORY_STRIDE,
            )
            if step < REFLOW_STEPS - 1:
                print(
                    "Generating new target distribution for the next reflow step..."
                )
                with torch.no_grad():
                    new_pi_1_path = sample(
                        rectified_flow,
                        NUM_SAMPLING_STEPS,
                        current_pi_0.to(device),
                    )
                    current_pi_1 = new_pi_1_path[-1].cpu()
                print("New target generated.")
        print("\n--- 2-Rectified Flow Training Complete ---")

    # --- Part 4: MNIST Generation ---
    elif part == "4":
        print("\n--- Running Part 4: MNIST Generation ---")

        # Load MNIST dataset
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
            ]
        )

        mnist_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )

        mnist_loader = DataLoader(
            mnist_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        # Initialize model
        print("Initializing U-Net model...")
        rectified_flow = SimpleUNet(in_channels=1, time_emb_dim=32).to(device)
        optimizer = torch.optim.AdamW(
            rectified_flow.parameters(), lr=2e-4, weight_decay=0.0
        )

        # Train
        print("Training on MNIST...")
        train_rectified_flow_mnist(
            rectified_flow,
            optimizer,
            mnist_loader,
            NB_EPOCHS=20,  # Use uppercase to match function parameter
            device=device,
        )

        # Generate samples T = 100
        print("Generating MNIST samples...")
        rectified_flow.eval()
        samples = sample_mnist(
            rectified_flow, T=100, batch_size=64, device=device
        )

        # Generate samples T = 1
        samples_t1 = sample_mnist(
            rectified_flow, T=1, batch_size=64, device=device
        )

        # Generate samples T = 5
        samples_t5 = sample_mnist(
            rectified_flow, T=5, batch_size=64, device=device
        )

        # Visualize
        visualize_mnist_samples(samples, "./images/mnist_generated.png", nrow=8)
        visualize_mnist_samples(
            samples_t1, "./images/mnist_generated_t1.png", nrow=8
        )
        visualize_mnist_samples(
            samples_t5, "./images/mnist_generated_t5.png", nrow=8
        )

        # Also save some real samples for comparison
        real_samples, _ = next(iter(mnist_loader))
        visualize_mnist_samples(
            real_samples[:64], "./images/mnist_real.png", nrow=8
        )

        print("\n--- MNIST Generation Complete ---")
