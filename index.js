var ctx = document.getElementById("myChart");
var modelname = "2022-05-15-15-13-26";
var cancelled = false;

var arr = "./src/Models".split("/");
var last = arr[arr.length - 1] || arr[arr.length - 2];
var dataset;
var mouseMove = false;
var prev = -1;
var running = false;

fetch("./src/Models/" + modelname + "/scores.json")
  .then((response) => response.json())
  .then((data) => {
    dataset = data;
    makeChart(data);
  });

function makeChart(data) {
  var myChart = new Chart(ctx, {
    type: "line",
    data: {
      recording: data["Recording"],
      labels: data["Iteration"],
      datasets: [
        {
          label: "Intrinsic Reward",
          data: data["Intrinsic Reward"],
          backgroundColor: "rgba(255,0,0,0.7)",
          borderColor: "rgba(255,0,0,0.7)",
        },
        {
          label: "RND Loss",
          data: data["RND Loss"],
          backgroundColor: "blue",
          borderColor: "blue",
        },
      ],
    },
    options: {
      plugins: {
        tooltip: {
          enabled: true,
          position: "nearest",
          backgroundColor: "rbga(0,0,0,0)",
          callbacks: {title: function(){return}}
        },
      },
      interaction: {
        intersect: false,
        mode: "index",
      },
    },
    plugins: [customTooltip],
  });
}

const customTooltip = {
  id: "customTooltip",
  afterDraw(chart, args, options) {
    if (chart.tooltip?._active?.length) {
      var canvas = document.createElement("canvas");
      const activePoint = chart.tooltip._active[0];
      const datapoint = activePoint.index;
      const datasetIndex = activePoint.index;
      const { ctx } = chart;
      const sizeGif = 180;
      recording = chart.data.recording[datasetIndex];

      ctx.fillStyle = "rgba(10,10,10,0.3)";

      // draw gif
      mouseMove = datasetIndex != prev;
      if (mouseMove) {
        prev = datasetIndex;

        plotRecording(canvas, chart, recording, datasetIndex, sizeGif);
      }

      //draw vertical line

      let x = chart.tooltip._active[0].element.x;
      let yAxis = chart.scales.y;
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(x, yAxis.top);
      ctx.lineTo(x, yAxis.bottom);
      ctx.lineWidth = 1;
      ctx.strokeStyle = "#ff0000";
      ctx.stroke();
      ctx.restore();
    }
  },
};

var imgCanvas = document.getElementById("imageCanvas");

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function plotRecording(canvas, chart, recording, dataIndex, sizeGif) {
  for (let frame of recording) {
    plotFrame(frame, canvas, chart, sizeGif);
    mouseMove = dataIndex != prev;
    if (mouseMove) {
      return;
    }
    await sleep(10);
  }

  prev = dataIndex;
}

function plotFrame(frame, canvas, chart, sizeGif) {
  // ctx = imgCanvas.getContext('2d');
  const pointX = chart._active[1].element.x;
  const pointY = chart._active[1].element.y;
  const offset = 8;

  // console.log(pointX, pointY);

  canvas.width = recording[0][0][0].length;
  canvas.height = recording[0][0][1].length;

  var idata = chart.ctx.createImageData(canvas.width, canvas.height);
  frame = frame.flat(2);

  for (var i = 0; i < frame.length; i++) {
    //grayscale to rgba
    idata.data[4 * i] = frame[i];
    idata.data[4 * i + 1] = frame[i];
    idata.data[4 * i + 2] = frame[i];
    idata.data[4 * i + 3] = 255; // not changing the transparency
  }
  const temp = createImageBitmap(idata).then(
    function (value) {
      var x, y;
      if (pointX > chart.width /2) {
        x = pointX- offset - sizeGif;
      } else {
        x = pointX + offset;
      }
      if (pointY < sizeGif) {
        y = chart.tooltip.y + chart.tooltip.height + offset;
      } else {
        y = chart.tooltip.y - offset - sizeGif;
      }

      chart.ctx.drawImage(value, x, y, sizeGif, sizeGif);
    },
    function (error) {}
  );
  // plot it left or right from datapoint

  //   if (pointX > chart.width - canvas.width) {
  //     // ctx.putImageData(
  //     //   idata,
  //     //   pointX - offset - canvas.width,
  //     //   pointY - canvas.height - offset
  //     // );
  //     ctx.drawImage(temp,pointX - offset - canvas.width, pointY - canvas.height - offset);
  //   } else {
  //     ctx.drawImage(temp, pointX + offset, pointY - canvas.height - offset);

  // //    ctx.putImageData(idata, pointX + offset, pointY - canvas.height - offset);
  //   }
}

// }
const background = () => {
  ctx.fillStyle = "##ff9505";
  ctx.fillRect(0, 0, size, size); // fill the entire canvas
};

// callbacks: {
//   title: function (context) {
//     // just for the gif
//     mouseMove = dataset["Recording"][context[0].dataIndex] != prev;
//     if (mouseMove) {
//       prev = dataset["Recording"][context[0].dataIndex];
//       callerFun(context);
//     }
//     return "PG Loss " + dataset['PG Loss'][context[0].dataIndex];
//   },
// },

CanvasRenderingContext2D.prototype.roundRect = function (x, y, w, h, r) {
  if (w < 2 * r) r = w / 2;
  if (h < 2 * r) r = h / 2;
  this.beginPath();
  this.moveTo(x+r, y);
  this.arcTo(x+w, y,   x+w, y+h, r);
  this.arcTo(x+w, y+h, x,   y+h, r);
  this.arcTo(x,   y+h, x,   y,   r);
  this.arcTo(x,   y,   x+w, y,   r);
  this.closePath();
  return this;
}