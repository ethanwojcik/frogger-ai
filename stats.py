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

    def update():
        while(True):
            stats = stats_queue.get()
            print(stats)
            episodes.append(stats['episode'])
            avg_high_scores.append(stats['avg_high_score'])
            rolling_high_scores.append(stats['avg_high_score'])
            effective_percentages.append(stats['effective_percentage'])
        while not stats_queue.empty():
            stats = stats_queue.get()
            print(stats)
            #episodes.append(stats['episode'])
            #avg_high_scores.append(stats['avg_high_score'])
           # rolling_high_scores.append(stats['avg_high_score'])
           # effective_percentages.append(stats['effective_percentage'])


    
    update()