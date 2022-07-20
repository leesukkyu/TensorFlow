import tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-node'
import path from 'path'

// 1. 과거의 데이터를 준비합니다.
var 온도 = [1, 20, 30]
var 판매량 = [1, 2, 3]
var 원인 = tf.tensor(온도)
var 결과 = tf.tensor(판매량)
console.log(온도.length)
console.log(판매량.length)
// 2. 모델의 모양을 만듭니다.
// var X = tf.input({ shape: [1] });
// var Y = tf.layers.dense({ units: 1 }).apply(X);
// var model = tf.model({ inputs: X, outputs: Y });
// var compileParam = { optimizer: tf.train.adam(), loss: tf.losses.meanSquaredError }
// model.compile(compileParam);

var inputs = tf.input({shape: [1]})
var outputs = tf.layers.dense({units: 1}).apply(inputs)
var model = tf.model({inputs, outputs})
model.compile({optimizer: tf.train.adam(), loss: tf.losses.meanSquaredError})

// 3. 데이터로 모델을 학습시킵니다.
// var fitParam = { epochs: 100}
// model.fit(원인, 결과, fitParam).then(function (result) {

// 4. 모델을 이용합니다.
// 4.1 기존의 데이터를 이용
// var 예측한결과 = model.predict(원인);
// 예측한결과.print();

// });
var fitParam = {epochs: 30}
const run = async () => {
    await model.fit(원인, 결과, fitParam).then((result) => {
        console.log('학습 끝')
    })
    // 4.2 새로운 데이터를 이용
    var 다음주온도 = [1, 20, 30]
    model.predict(tf.tensor(다음주온도)).print()
    await model.save(`file://${path.resolve('path')}/my-model`)
}

run()
