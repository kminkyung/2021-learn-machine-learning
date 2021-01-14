const tf = require('@tensorflow/tfjs-node');

/*
// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// Prepare the model for training: Specify the loss and the optimizer.
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// Train the model using the data.
model.fit(xs, ys).then(() => {
  // Use the model to do inference on a data point the model hasn't seen before:
  model.predict(tf.tensor2d([5], [1, 1])).print();
});
*/

let temperatures = [20, 21, 22, 23];
let sales = [40, 42, 44, 46];
const cause = tf.tensor(temperatures);
const effect = tf.tensor(sales);

const X = tf.input({ shape: [1] });
const Y = tf.layers.dense({ units: 1 }).apply(X);
const model = tf.model({ inputs: X, outputs: Y });
const compileParam = { optimizer: tf.train.adam(), loss: tf.losses.meanSquaredError }
model.compile(compileParam);

const fitParam = { epochs: 100 }; // epochs: 트레이닝 횟수
model.fit(cause, effect, fitParam).then(result => {
  let nextWeekTemp = [15, 16, 17, 18, 19];
  let nextWeekCause = tf.tensor2d(nextWeekTemp, [nextWeekTemp.length, 1]);
  let nextWeekEffect = model.predict(nextWeekCause);
  nextWeekEffect.print();
})