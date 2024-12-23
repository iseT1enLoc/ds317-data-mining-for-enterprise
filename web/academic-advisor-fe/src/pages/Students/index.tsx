import MUIDataTable from "mui-datatables";
import { useEffect, useState } from "react";
import axios, { AxiosResponse } from "axios";

// Kiểu dữ liệu cho điểm trung bình của sinh viên
interface AverageScore {
  _id: string;
  studentId: string;
  averageScore: number; // Có thể là số âm, cần xử lý cẩn thận
  semester: string;
  academicYear: string;
}

// Kiểu dữ liệu cho điểm rèn luyện của sinh viên
interface TrainingPoint {
  _id: string;
  studentId: string;
  trainingPoint: number; // Điểm rèn luyện (có thể là từ 0-100)
  semester: string;
  academicYear: string;
}

// Kiểu dữ liệu chính cho thông tin sinh viên
interface Student {
  _id: string;
  studentId: string;
  name: string;
  admissionMethod: string;
  gender: "Male" | "Female" | string; // Giới tính có thể thêm các giá trị khác nếu cần
  faculty: string; // Khoa
  educationSystem: string; // Hệ đào tạo
  majorCode: string; // Mã ngành
  admissionScore: number; // Điểm xét tuyển
  placeOfBirth: string; // Nơi sinh
  phoneNumber: string; // Số điện thoại
  email: string; // Email liên lạc
  averageScores: AverageScore[]; // Danh sách điểm trung bình
  trainingPoints: TrainingPoint[]; // Danh sách điểm rèn luyện
}

interface statusData {
  message: string;
  data: string[];
}

const columns = [
  "MSSV",
  "Họ tên",
  "Giới tính",
  "Nơi sinh",
  "Điện thoại",
  "Email",
];

function Students(): JSX.Element {
  const [students, setStudents] = useState<Student[]>([]); // State kiểu mảng các đối tượng Student
  const [data, setData] = useState<string[][]>([]); // State để lưu trữ dữ liệu cho bảng
  const [statusData, setStatusData] = useState<statusData>(); // State để lưu trữ dữ liệu trạng thái từ API

  useEffect(() => {
    axios
      .get<Student[]>("http://localhost:3000/api/students") // Gọi API lấy thông tin sinh viên
      .then((response: AxiosResponse<Student[]>) => {
        setStudents(response.data); // Cập nhật danh sách sinh viên
      })
      .catch((error) => {
        console.error("Error fetching students:", error);
      });
  }, []);

  useEffect(() => {
    // Tạo dữ liệu cho bảng dựa trên danh sách students
    const formattedData = students.map((student) => [
      student.studentId, // MSSV
      student.name, // Họ tên
      student.gender, // Giới tính
      student.placeOfBirth, // Nơi sinh
      student.phoneNumber, // Điện thoại
      student.email, // Email
    ]);

    setData(formattedData); // Cập nhật lại dữ liệu bảng
  }, [students]);

  // Hàm gọi API để lấy điểm trạng thái của sinh viên
  const fetchStatusData = () => {
    axios
      .get<statusData>("http://localhost:3000/api/students-to-csv")
      .then((response: AxiosResponse<statusData>) => {
        setStatusData(response.data); // Lưu dữ liệu trạng thái
      })
      .catch((error) => {
        console.error("Error fetching status data:", error);
      });
  };

  useEffect(() => {
    if (statusData && statusData.data.length > 0) {
      // Mảng các giá trị trạng thái
      const statusLabels = ["Giỏi/xuất sắc", "Khá", "Trung bình khá", "Trung bình", "Chưa đạt"];
  
      // Ghép giá trị từ statusData vào phần tử cuối cùng của mỗi mảng con trong data
      const updatedData = data.map((row, index) => {
        if (index < statusData.data.length) {
          const statusIndex = statusData.data[index]; // Lấy giá trị trạng thái
          const statusLabel = statusLabels[Number(statusIndex)] || "Chưa xác định"; // Quy đổi thành nhãn
          return [...row, statusLabel]; // Thêm nhãn trạng thái vào mảng con
        }
        return row;
      });
  
      setData(updatedData); // Cập nhật lại dữ liệu bảng với trạng thái mới
    }
  }, [statusData]);

  const options = {
    filterType: "checkbox",
  };

  return (
    <div className="mt-10 mx-auto pb-4">
      <button
        onClick={fetchStatusData}
        className="mb-4 p-2 bg-blue-500 text-white rounded"
      >
        Lấy dữ liệu dự đoán
      </button>
      <div className="flex justify-center">
        <MUIDataTable
          title={"Danh sách sinh viên"}
          data={data}
          columns={[...columns, "Trạng thái"]} // Thêm cột "Trạng thái"
          // options={options}
        />
      </div>
    </div>
  );
}

export default Students;
