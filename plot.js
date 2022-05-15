var ctx = document.getElementById("myChart");
var modelname = "2022-05-15-15-13-26";

var arr = "./src/Models".split("/");
var last = arr[arr.length - 1] || arr[arr.length - 2];
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
  console.log(data);

  var myChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: data["Iteration"],
      datasets: [
        {
          label: "Intrinsic Reward",
          data: data["Intrinsic Reward"],
        },
        {
          label: "RND Loss",
          data: data["RND Loss"],
        },
      ],
    },
    options: {
      interaction: {
        intersect: false,
        mode: "index",
      },
      plugins: {
        tooltip: {
          filter: function (tooltipItem) {
            return tooltipItem.datasetIndex === 0;
          },

          callbacks: {
            title: function (context) {
              callerFun(context);
            },
          },

          //   backgroundColor: '#000000',
          //   titleFontSize: 16,
          //   titleFontColor: '#0066ff',
          //   bodyFontColor: '#000',
          //   bodyFontSize: 14,
          //   displayColors: false
        },
      },
    },
    plugins: [
      {
        afterDraw: (chart) => {
          if (chart.tooltip?._active?.length) {
            let x = chart.tooltip._active[0].element.x;
            let yAxis = chart.scales.y;
            let ctx = chart.ctx;
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
      },
    ],
  });
}

var myImg = document.getElementById("img");

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function callerFun(context) {
  mouseMove = dataset["Recording"][context[0].dataIndex] != prev;
  if (mouseMove){
    plotRecording(context);
    prev = dataset["Recording"][context[0].dataIndex]
  }
}

async function plotRecording(context) {
  var canvas = document.createElement("canvas");
  var ctx = canvas.getContext("2d");
  recording = dataset["Recording"][context[0].dataIndex];

  canvas.width = recording[0][0][0].length;
  canvas.height = recording[0][0][1].length;
  mouseMove = context[0].dataIndex != prev;

  for (let frame of recording) {
    console.log(context[0].dataIndex);
    child = myImg.firstChild;

    if (child != null) {
      child.remove();
    }

    var idata = ctx.createImageData(canvas.width, canvas.height);
    frame = frame.flat(2);

    for (var i = 0; i < frame.length; i++) {
      //grayscale to rgba
      idata.data[4 * i] = frame[i];
      idata.data[4 * i + 1] = frame[i];
      idata.data[4 * i + 2] = frame[i];
      idata.data[4 * i + 3] = 255; // not changing the transparency
    }

    ctx.putImageData(idata, 0, 0);
    var image = new Image();
    image.src = canvas.toDataURL();
    myImg.appendChild(image);

    await sleep(10);
  }
}
const background = () => {
  ctx.fillStyle = "##ff9505";
  ctx.fillRect(0, 0, size, size); // fill the entire canvas
};
