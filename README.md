# Machine Learning Games 🧠

An interactive field guide to the most famous machine-learning algorithms — each one
turned into a small game you can play with in your browser. Drag points, place
clusters, watch a network learn, train an agent. Every algorithm shows the live
simulation first, then explains *exactly how it works* underneath.

Everything runs locally in your browser. Nothing is uploaded; there is no backend.

## The games

| Family | Algorithm | What you do |
| --- | --- | --- |
| Foundations | **Gradient Descent** | Roll a ball down a loss surface; tune the learning rate and watch it converge — or explode. |
| Foundations | **Overfitting & Regularization** | Crank a polynomial's degree and watch train error fall while test error U-turns. |
| Regression | **Linear Regression** | Drag data points and watch the best-fit line chase them via least squares. |
| Regression | **Gradient Boosting** | Stack shallow trees in sequence, each fixing the last one's residuals. |
| Classification | **Logistic Regression** | Paint two classes and watch a probability boundary form. |
| Classification | **k-Nearest Neighbours** | Move a query point and let its neighbours vote on its label. |
| Classification | **Decision Tree** | Watch greedy axis-aligned splits carve the plane by impurity. |
| Classification | **Random Forest** | Pile up bootstrapped trees and watch their votes smooth into one boundary. |
| Classification | **Support Vector Machine** | Find the widest possible street between two classes. |
| Classification | **Naive Bayes** | A live spam filter that shows the probability maths word by word. |
| Classification | **Neural Network** | A tiny multi-layer perceptron learning curved boundaries by backprop. |
| Clustering | **k-Means** | Drop points and watch centroids hunt for the groups. |
| Clustering | **DBSCAN** | Cluster by density — arbitrary shapes, automatic count, outliers flagged. |
| Dimensionality | **PCA** | Find the directions a cloud of data actually varies in. |
| Reinforcement | **Q-Learning** | Train an agent to cross a grid world by trial, error and reward. |
| Reinforcement | **Multi-Armed Bandit** | Balance explore vs. exploit across slot machines with ε-greedy, UCB and Thompson. |
| Computer vision | **Convolution** | Slide kernels over an image to pull out edges, blurs and textures — the heart of CNNs. |

## Running it

This is a static site built with [Vite](https://vitejs.dev/).

```bash
npm install
npm run dev      # local dev server with hot reload
npm run build    # outputs a fully static bundle to dist/
npm run preview  # serve the built bundle
```

> The site uses native ES modules, so it should be served over HTTP (any static
> server works) rather than opened directly from the file system.

## How it's built

- `index.html` — the app shell (sidebar, mobile drawer, mount points).
- `styles.css` — the whole design system.
- `shared.js` — the shared toolkit every game uses: a `Plot` for 2-D canvases,
  a `Loop` for animation, dataset generators, colours and small maths helpers.
- `main.js` — the registry-driven router. It reads the list of games, builds the
  navigation, dashboard and view sections, and lazily mounts each game.
- `games/` — one self-contained module per algorithm. Each default-exports its
  metadata, its lesson, and a `mount(root)` function that builds the game.

Adding a new game is just: drop a file in `games/`, follow the contract in
`games/_contract.md`, and register it in `games/index.js`.
