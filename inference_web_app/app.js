
const get_prediction = async function(x_val) {
  const MODEL_URL = '2018_10_03_18_44_38/tensorflowjs_model.pb';
  const WEIGHTS_URL = '2018_10_03_18_44_38/weights_manifest.json';

  console.log('hello');

  const model = await tf.loadFrozenModel(MODEL_URL, WEIGHTS_URL);
  const y = tf.squeeze(model.predict({Placeholder:tf.tensor([x_val])})).dataSync()[0];
  const y_div = document.getElementById('y');
  y_div.innerHTML = y;
}


document.getElementById('input-form').addEventListener('submit',
  function(e) {
    const x = Number(document.getElementById('x').value);
    get_prediction(x);

    e.preventDefault();
  });