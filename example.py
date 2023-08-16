import local_face.helpers.datasets as datasets
from local_face.local_face import *
from local_face.helpers.plotters import *
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KernelDensity
import time
import warnings

warnings.filterwarnings("ignore")
graph = False
# the factual point
factual = np.array([0.04, 0.375])
# parameters for locating counterfactual and path
k = 5
thresh = 0.95
dist = 0.1
method_type = 'strict'
# parameters for data generation
seed = 10
samples = 300
noise = 0.15
# parameters for model creation
band_width = 0.025

# create data
generator = datasets.create_two_moons  # get data generator
data = generator(samples, seed, noise)  # generate data
X = data.iloc[:, :2]  # ensure data is 2d
y = data.y  # get target

con_X = datasets.constrain_search(X, [[], []])
con_X['y'] = data.y

# train model and density estimator using training data
model = MLPClassifier(hidden_layer_sizes=(14, 11, 8), random_state=1, max_iter=700).fit(X, y)
dense = KernelDensity(kernel='gaussian', bandwidth=band_width).fit(X)
X1 = data[["x1", "x2"]]

start_total = time.time()
# initialise Local-FACE model
face = LocalFace(X1, model, dense)

# find a counterfactual
print(r"Finding counterfactual x' (explore)...")
start_explore = time.time()
steps, cf = face.find_cf(factual, k=k, thresh=thresh, mom=3)
print('Explore time taken: {} seconds'.format(np.round(time.time() - start_explore, 2)))
print('---------------------------------')

# generate graph nodes through data from factual to counterfactual
print(r'Creating graph G (Exploit)...')
start_exploit = time.time()
best_steps, G = face.generate_graph(factual, cf, k, thresh, 1, 10, early=True, method=method_type)
# create edges between viable points and calculate the weights
"""prob = face.dense.score([factual])
G = face.create_edges(1, 10, method='strict')"""
print('Exploit time taken: {} seconds'.format(np.round(time.time() - start_exploit, 2)))
print('---------------------------------')

# calculate shortest path through G from factual to counterfactual
print(r"Finding shortest path from x to x' through G (Enhance)...")
start_enhance = time.time()
shortest_path = face.shortest_path(method=method_type)
print('Enhance time taken: {} seconds'.format(np.round(time.time() - start_enhance, 2)))
print('---------------------------------')
print('Total time taken: {} seconds'.format(np.round(time.time() - start_total, 2)))

# plotting procedure
# plot the data, the decision function, the searched path
if not graph:
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), sharey=True)
    # finding counterfactual
    ax[0] = plot_dataset(ax[0], con_X)
    ax[0] = plot_decision_boundary(ax[0], X1, model)
    ax[0].plot(factual[0], factual[1], 'go', label='factual')
    ax[0].plot(steps[:, 0], steps[:, 1], '-y', label='Explore', linewidth=3)
    ax[0].set_xlim([0, 0.5])
    ax[0].set_ylim([0.2, 0.9])
    ax[0].title.set_text('Explore')

    ax[1] = plot_density(ax[1], X1, dense, levels=5)
    ax[1] = plot_graph(ax[1], X1, model, best_steps, G, shortest_path)
    ax[1].set_xlim([0, 0.5])
    ax[1].set_ylim([0.2, 0.9])
    # ax[1].plot(best_steps[:, 0], best_steps[:, 1], '-k', label='Exploit', linewidth=2.5)
    # ax[1].plot(factual[0], factual[1], 'go', label='factual')
    ax[1].plot(best_steps[-1, 0], best_steps[-1, 1], '*b', label='$x^\prime$', markersize=12, alpha=0.7, markeredgecolor='white')
    ax[1].title.set_text('Exploit')

    ax[2] = plot_density(ax[2], X1, dense, levels=5, alpha=0.4, over=True)
    ax[2] = plot_decision_boundary(ax[2], X1, model, alpha=0.6)
    ax[2].set_xlim([0, 0.5])
    ax[2].set_ylim([0.2, 0.9])
    ax[2].plot(best_steps[shortest_path, 0], best_steps[shortest_path, 1], '-g', label='Enhance', linewidth=2)
    ax[2].plot(factual[0], factual[1], 'go', label='factual')
    ax[2].plot(best_steps[-1, 0], best_steps[-1, 1], '*b', label='$x^\prime$', markersize=12, alpha=0.7, markeredgecolor='white')
    ax[2].plot([factual[0], best_steps[-1, 0]],
               [factual[1], best_steps[-1, 1]], '-r', linewidth=2)
    ax[2].plot([best_steps[shortest_path[1], 0], best_steps[shortest_path[3], 0]],
               [best_steps[shortest_path[1], 1], best_steps[shortest_path[3], 1]], '-r', label='illegal edge', linewidth=2)
    ax[2].title.set_text('Enhance')
    ax[2].annotate("", xy=(0.21, 0.375), xytext=(0.21, 0.45), arrowprops=dict(arrowstyle="->", lw=2.5, color='black', alpha=.7))
    ax[2].text(0.21, 0.475, 'not allowed', va='center', ha='center',
               rotation='horizontal', fontsize=12, color='black', alpha=.7)
    ax[2].annotate("", xy=(0.19, 0.68), xytext=(0.21, 0.5), arrowprops=dict(arrowstyle="->", lw=2.5, color='black', alpha=.7))
    ax[2].annotate("", xy=(0.31, 0.665), xytext=(0.21, 0.5), arrowprops=dict(arrowstyle="->", lw=2.5, color='black', alpha=.7))

    fig.tight_layout()
    plt.savefig('local_face/plots/paper_figure_strict_graph.png', format='png')
    plt.show()

if graph:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), sharey=True)
    ax = plot_density(ax, X1, dense, levels=5)
    ax = plot_graph(ax, X1, model, best_steps, G, shortest_path)
    ax = plot_dataset(ax, data)
    ax.set_xlim([0, 0.5])
    ax.set_ylim([0.2, 0.85])
    plt.savefig('local_face/plots/network_dense{}.png'.format(factual), format='png', dpi=400)
    plt.show()
print('stop')
