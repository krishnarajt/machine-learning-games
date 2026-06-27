// Game registry. Order here = order in the sidebar and dashboard launcher
// (roughly simple → advanced, grouped by family). Each module default-exports
// a game object (see games/_contract.md).
import gradientDescent from "./gradient-descent.js";
import overfitting from "./overfitting.js";
import linearRegression from "./linear-regression.js";
import gradientBoosting from "./gradient-boosting.js";
import logisticRegression from "./logistic-regression.js";
import knn from "./knn.js";
import decisionTree from "./decision-tree.js";
import randomForest from "./random-forest.js";
import svm from "./svm.js";
import naiveBayes from "./naive-bayes.js";
import neuralNetwork from "./neural-network.js";
import kmeans from "./kmeans.js";
import dbscan from "./dbscan.js";
import pca from "./pca.js";
import qLearning from "./q-learning.js";
import bandit from "./bandit.js";
import convolution from "./convolution.js";

export const GAMES = [
  gradientDescent,
  overfitting,
  linearRegression,
  gradientBoosting,
  logisticRegression,
  knn,
  decisionTree,
  randomForest,
  svm,
  naiveBayes,
  neuralNetwork,
  kmeans,
  dbscan,
  pca,
  qLearning,
  bandit,
  convolution,
];
