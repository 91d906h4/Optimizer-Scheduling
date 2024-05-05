import matplotlib.pyplot as plt


class Figure:
    def __init__(self, figsize: tuple=(16, 4)) -> None:
        self.figsize        = figsize

    def draw(self, title: str, loss_history: list, acc_history: list, iter_num: int, epoch_num: int, save: bool=True) -> None:
        temp_x = [iter_num * i for i in range(epoch_num+1)]

        # Create figure.
        figure = plt.figure(figsize=self.figsize)

        # Plot loss history.
        ax = figure.add_subplot(1, 2, 1)

        plt.title("Train Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")

        # Plot loss.
        ax.plot(loss_history, label="Loss")

        # Calculate and plot average loss.
        avg_loss_x = [(temp_x[i] + temp_x[i+1]) // 2 for i in range(epoch_num)]
        avg_loss_y = [sum(loss_history[temp_x[i]:temp_x[i+1]]) / iter_num for i in range(epoch_num)]
        ax.plot(avg_loss_x, avg_loss_y, 'r', label="Average Loss")
        ax.legend()


        # Plot accuracy history.
        ax = figure.add_subplot(1, 2, 2)

        plt.title("Train Accuracy")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy (%)")

        # Plot accuracy.
        ax.plot(acc_history, label="Accuracy")

        # Calculate and plot average accuracy.
        avg_acc_x = [(temp_x[i] + temp_x[i+1]) // 2 for i in range(epoch_num)]
        avg_acc_y = [sum(acc_history[temp_x[i]:temp_x[i+1]]) / iter_num for i in range(epoch_num)]
        ax.plot(avg_acc_x, avg_acc_y, 'r', label="Average Accuracy")
        ax.legend()

        # Set title.
        if title:
            plt.suptitle(t=title, x=0.5, y=0, ha="center", fontsize=12)

        # Show figure.
        plt.show()

        # Save image.
        if save:
            figure.savefig(fname=f"./{title}.png")