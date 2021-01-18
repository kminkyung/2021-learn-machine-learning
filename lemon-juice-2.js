const tf = require('@tensorflow/tfjs-node');

let temperatures = [20, 21, 22, 23];
let sales = [40, 42, 44, 46];
const cause = tf.tensor(temperatures);
const effect = tf.tensor(sales);

const X = tf.input({ shape: [1] });
const Y = tf.layers.dense({ units: 1 }).apply(X);
const model = tf.model({ inputs: X, outputs: Y });
const compileParam = { optimizer: tf.train.adam(), loss: tf.losses.meanSquaredError }
model.compile(compileParam);

const fitParam = {
  epochs: 100,
  callbacks: {
    onEpochEnd: (epoch, logs) => {
      // console.log('epoch', epoch, logs, 'RMSE =>', Math.sqrt(logs.loss))
    }
  }
};

model.fit(cause, effect, fitParam).then(result => {
  const prediction = model.predict(cause);
  prediction.print();
})
