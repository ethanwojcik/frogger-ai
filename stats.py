import matplotlib.pyplot as plt
import matplotlib.animation as animation
import multiprocessing as mp

def stats_plotter(stats_queue):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.title("Frogger Training Stats")
    plt.xlabel("Episode")
    plt.ylabel("Score / Effectiveness")

    episodes = []
    avg_high_scores = []
    rolling_high_scores = []
    effective_percentages = []

    def update(frame):
        while not stats_queue.empty():
            stats = stats_queue.get()
            episodes.append(stats['episode'])
            avg_high_scores.append(stats['avg_high_score'])
            rolling_high_scores.append(stats['rolling_high_score'])
            effective_percentages.append(stats['effective_percentage'])

        ax.clear()
        ax.plot(episodes, avg_high_scores, label="Avg High Score")
        ax.plot(episodes, rolling_high_scores, label="Rolling High Score")
        ax.plot(episodes, effective_percentages, label="Effective %")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Score / Effectiveness")
        ax.legend()
        ax.set_title("Frogger Training Stats")
        ax.grid(True)

    ani = animation.FuncAnimation(fig, update, interval=500)
    
    plt.show(ani)