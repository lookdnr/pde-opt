from dataclasses import dataclass, field
import matplotlib.pyplot as plt

@dataclass
class OptimisationMonitor:
    history: list = field(default_factory=list)
    exit_reason: str = field(default_factory=str)

    def record(self, details: dict) -> None:
        self.history.append(details)

    def show(self):
        if not self.history:
            raise ValueError("No history has been recorded.")

        iters = [entry["iter"] for entry in self.history]
        J_vals = [entry["J"] for entry in self.history]
        grad_norms = [entry["grad_norm"] for entry in self.history]

        _, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

        axes[0].plot(iters, J_vals, marker="o", linewidth=1.5, markersize=3, color="tab:blue")
        axes[0].set_title("Functional Convergence")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("J")
        axes[0].grid(True, alpha=0.3)

        axes[1].semilogy(iters, grad_norms, marker="o", linewidth=1.5, markersize=3, color="tab:orange")
        axes[1].set_title("Gradient Norm Convergence")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("||grad J||")
        axes[1].grid(True, which="both", alpha=0.3)

        plt.show()