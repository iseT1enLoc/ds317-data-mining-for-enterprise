import { BarChart } from "@mui/x-charts/BarChart";
import { PieChart } from "@mui/x-charts/PieChart";
import { LineChart, lineElementClasses } from "@mui/x-charts/LineChart";
import { FaPeopleLine } from "react-icons/fa6";
import Stack from "@mui/material/Stack";
import { Gauge } from "@mui/x-charts/Gauge";
const uData = [4000, 3000, 2000, 2780, 1890, 2390, 3490];
const pData = [2400, 1398, 9800, 3908, 4800, 3800, 4300];
const amtData = [2400, 2210, 0, 2000, 2181, 2500, 2100];
const xLabels = [
  "Page A",
  "Page B",
  "Page C",
  "Page D",
  "Page E",
  "Page F",
  "Page G",
];

function Dashboard() {
  return (
    <div className="bg-slate-100 w-full">
      <h2 className="font-bold text-xl text-black p-4">Advisor Dashboard</h2>
      <div className="flex p-4">
        <div className="w-64 h-24 bg-white flex items-center gap-x-2 rounded-md m-4">
          <div className="size-16 bg-green-200 rounded-full ml-3 flex items-center justify-center">
            <FaPeopleLine size={30} color="green" />
          </div>
          <hr className="border-t-2 border-red-600 w-2/12 rotate-90" />
          <div className="flex flex-col">
            <p className="text-black/30 text-xl">Students</p>
            <span className="font-bold text-xl">4.144</span>
          </div>
        </div>
        <div className="w-64 h-24 bg-white flex items-center gap-x-2 rounded-md m-4">
          <div className="size-16 bg-green-200 rounded-full ml-3 flex items-center justify-center">
            <FaPeopleLine size={30} color="green" />
          </div>
          <hr className="border-t-2 border-red-600 w-2/12 rotate-90" />
          <div className="flex flex-col">
            <p className="text-black/30 text-xl">Students</p>
            <span className="font-bold text-xl">4.144</span>
          </div>
        </div>
        <div className="flex items-center">
          <Stack
            direction={{ xs: "column", md: "row" }}
            spacing={{ xs: 1, md: 3 }}
          >
            <Gauge width={100} height={100} value={60} />
          </Stack>
        </div>
      </div>
      <div className="flex justify-around items-center">
        <div>
          <BarChart
            xAxis={[
              { scaleType: "band", data: ["group A", "group B", "group C"] },
            ]}
            series={[
              { data: [4, 3, 5] },
              { data: [1, 6, 3] },
              { data: [2, 5, 6] },
            ]}
            width={500}
            height={300}
          />
        </div>
        <div>
          <PieChart
            series={[
              {
                data: [
                  { id: 0, value: 10, label: "series A" },
                  { id: 1, value: 15, label: "series B" },
                  { id: 2, value: 20, label: "series C" },
                ],
              },
            ]}
            width={400}
            height={200}
          />
        </div>
      </div>
      <div className="flex justify-center items-center mx-auto">
        <LineChart
          width={800}
          height={300}
          series={[
            {
              data: uData,
              label: "uv",
              area: true,
              stack: "total",
              showMark: false,
            },
            {
              data: pData,
              label: "pv",
              area: true,
              stack: "total",
              showMark: false,
            },
            {
              data: amtData,
              label: "amt",
              area: true,
              stack: "total",
              showMark: false,
            },
          ]}
          xAxis={[{ scaleType: "point", data: xLabels }]}
          sx={{
            [`& .${lineElementClasses.root}`]: {
              display: "none",
            },
          }}
        />
      </div>
    </div>
  );
}

export default Dashboard;
