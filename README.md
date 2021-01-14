# Tensorflow.js 시작하기
[tensorflow.js](https://www.tensorflow.org/js/?hl=ko) 공식 페이지

### 지도학습/회귀
*Supervised learning/regression*
* 서버 환경
    ```shell script
      npm i @tensorflow/tfjs-node
    ```
    ```js
      const tf = require('@tensorflow/tfjs-node');
  
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
    ```

### MobileNet
*이미지 분류 모델*
1. 브라우저 환경
    ```html
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.0.1"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@1.0.0"></script>
   
   <img id="img" src="cat.jpg">
   
   <script>
     const img = document.getElementById('img');
     mobilenet.load().then(model => {
       model.classify(img).then(predictions => {
            console.log('Predictions: ');
            console.log(predictions);
          });
        });
    }
    </script>  
    }
    ```
   
2. 서버 환경
    ```shell script
    npm i @tensorflow-models/mobilenet
    ```
   ```js
   // import @tensorflow/tfjs 필요 없음
     const mobilenet = require('@tensorflow-models/mobilenet');
     
     const img = document.getElementById('img');
     
     const model = await mobilenet.load();
     
     const predictions = await model.classify(img);
     
     console.log('Predictions: ');
     console.log(predictions);
    ```
   
   