const Hapi = require('@hapi/hapi');
const Path = require('path');
const tf = require('@tensorflow/tfjs-node');
const { promisify } = require('util');
const fs = require('fs');
const readFileAsync = promisify(fs.readFile);

const init = async () => {
  const server = Hapi.server({
    port: 3001,
    host: 'localhost'
  });

  await server.register(require('@hapi/inert'));

  const loadModel = async () => {
    const modelPath = Path.join(__dirname, 'model', 'model.json');
    const model = await tf.loadLayersModel(`file://${modelPath}`);
    return model;
  };

  server.route({
    method: 'POST',
    path: '/predict',
    options: {
      payload: {
        output: 'stream',
        parse: true,
        multipart: true
      }
    },
    handler: async (request, h) => {
      const { image } = request.payload;
      if (!image) {
        return h.response('No files were uploaded.').code(400);
      } else if (
        image.hapi.headers['content-type'] !== 'image/jpeg' &&
        image.hapi.headers['content-type'] !== 'image/png'
      ) {
        return h.response('Invalid file type.').code(400);
      } else {
        const model = await loadModel();
        const imgBuffer = await readFileAsync(image.path);
        const img = tf.node.decodeImage(imgBuffer, 3);
        const resized = tf.image.resizeBilinear(img, [150, 150]);
        const expanded = resized.expandDims(0);
        const prediction = await model.predict(expanded).data();
        const result = prediction[0] > 0.5 ? 'Unhealthy' : 'Healthy';

        tf.dispose(img);
        tf.dispose(resized);
        tf.dispose(expanded);

        return { result };
      }
    }
  });

  await server.start();
  console.log(`Server listening on port ${server.info.port}`);
};

process.on('unhandledRejection', (err) => {
  console.log(err);
  process.exit(1);
});

init();
