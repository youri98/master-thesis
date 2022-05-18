const modelname = "2022-05-17-17-43-28";
const rolloutPerIteration = 128;

var ctx = document.getElementById("myChart");

var dataset;
var mouseMove = false;
var prev = -1;

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
          backgroundColor: "rgba(122, 222, 149,.7)",
          borderColor: "rgba(122, 222, 149,1)",
          // yAxisID: 'y',
          // xAxisID: 'x',

        },
        {
          label: "Extrinsic Reward",
          data: data["Extrinsic Reward"],
          backgroundColor: "rgba(107, 196, 214,0.7)",
          borderColor: "rgba(107, 196, 214,1 )",


        },
        {
          label: "RND Loss",
          data: data["RND Loss"],
          backgroundColor: "rgba(214, 47, 47, 0.7)",
          borderColor: "rgba(214, 47, 47, 1)",


        },
        {
          label: "Discriminator Loss",
          data: data["Discriminator Loss"],
          backgroundColor: "rgba(138, 62, 173, 0.7)",
          borderColor: "rgba(138, 62, 173, 0.7)",

        },
        {
          label: "Visited Rooms",
          data: data["Visited Rooms"],
          backgroundColor: "rgba(212, 137, 40, 0.7)",
          borderColor: "rgba(212, 137, 40, 0.7)",


        },
        {
          label: "Entropy",
          data: data["Entropy"],
          backgroundColor: "rgba(55, 50, 209, 0.7)",
          borderColor: "rgba(55, 50, 209, 0.7)",


        },
      ],
    },
    options: {
      scales: {
        x: {
          title: {
          display: true,
            text: "Frames"
        },
        ticks: {
          callback: function(value, index, ticks) {
              return rolloutPerIteration* value;
          }
        },
      },
        y: {
          title: {
          display: true,
            text: "Relative Scores"
        },
        }
      },
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
    title: {
      display: true,
      text: 'Chart Title',
    }

    // scales: {
    //   y: {
    //     type: 'linear',
    //     display: true,
    //     position: 'left',
    //   },
    //   y1: {
    //     type: 'linear',
    //     display: true,
    //     position: 'right',

    //     // grid line settings
    //     grid: {
    //       drawOnChartArea: false, // only want the grid lines for one axis to show up
    //     },
    //   },
    // }
  });
}

const customTooltip = {
  id: "customTooltip",
  afterDraw(chart, args, options) {
    if (chart.tooltip?._active?.length) {
      // console.log(chart.data);
      var canvas = document.createElement("canvas");
      const activePoint = chart.tooltip._active[0];
      const datapoint = activePoint.index;
      const datasetIndex = activePoint.index;
      const { ctx } = chart;
      const sizeGif = 180;
      recording = chart.data.recording[datasetIndex];

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
      ctx.setLineDash([5, 5]);
      ctx.strokeStyle = "#000000";
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
  const pointX = chart.tooltip.x;
  const pointY = chart.tooltip.y;
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
        x = pointX - offset;
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


// CanvasRenderingContext2D.prototype.roundRect = function (x, y, w, h, r) {
//   if (w < 2 * r) r = w / 2;
//   if (h < 2 * r) r = h / 2;
//   this.beginPath();
//   this.moveTo(x+r, y);
//   this.arcTo(x+w, y,   x+w, y+h, r);
//   this.arcTo(x+w, y+h, x,   y+h, r);
//   this.arcTo(x,   y+h, x,   y,   r);
//   this.arcTo(x,   y,   x+w, y,   r);
//   this.closePath();
//   return this;
// }