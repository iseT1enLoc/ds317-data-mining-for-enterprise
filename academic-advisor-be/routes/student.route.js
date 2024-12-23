// Import models
import fs from "fs";
import path from "path";
import { Parser } from "json2csv";

import { AverageSemesterScore } from "../models/semesterscores.model.js";
import { TrainingPoint } from "../models/trainingpoint.model.js";
import { StudentInfo } from "../models/studentinfo.model.js";

// Import authenticate and authorize middleware
// import { authenticate, authorizeRoles } from "../../middleware/user/authentication.js";

function webInitRouterStudent(app) {
  // app.use("/api/restaurant", authenticate, authorizeRoles("seller"));

  app.get("/api/students", async (req, res) => {
    try {
      const students = await StudentInfo.aggregate([
        {
          $lookup: {
            from: "averagesemesterscores",
            localField: "studentId",
            foreignField: "studentId",
            as: "averageScores",
          },
        },
        {
          $lookup: {
            from: "trainingpoints",
            localField: "studentId",
            foreignField: "studentId",
            as: "trainingPoints",
          },
        },
        {
          $addFields: {
            averageScores: {
              $sortArray: {
                input: "$averageScores",
                sortBy: {
                  academicYear: 1, // Sắp xếp tăng dần theo academicYear (string)
                  semester: 1, // Sắp xếp tăng dần theo semester (string hoặc number)
                },
              },
            },
            trainingPoints: {
              $sortArray: {
                input: "$trainingPoints",
                sortBy: {
                  academicYear: 1, // Sắp xếp tăng dần theo academicYear (string)
                  semester: 1, // Sắp xếp tăng dần theo semester (string hoặc number)
                },
              },
            },
          },
        },
      ]);
      res.status(200).json(students);
    } catch (error) {
      res
        .status(500)
        .json({ message: "Error fetching students' information", error });
    }
  });

  // API to get a single student's information by student ID
  app.get("/api/students/:studentId", async (req, res) => {
    const { studentId } = req.params;
    try {
      const student = await StudentInfo.aggregate([
        {
          $match: { studentId },
        },
        {
          $lookup: {
            from: "averagesemesterscores",
            localField: "studentId",
            foreignField: "studentId",
            as: "averageScores",
          },
        },
        {
          $lookup: {
            from: "trainingpoints",
            localField: "studentId",
            foreignField: "studentId",
            as: "trainingPoints",
          },
        },
      ]);

      if (!student.length) {
        return res.status(404).json({ message: "Student not found" });
      }

      res.status(200).json(student[0]);
    } catch (error) {
      res
        .status(500)
        .json({ message: "Error fetching student's information", error });
    }
  });

  // app.get("/api/students-to-csv", async (req, res) => {
  //   try {
  //     const students = await StudentInfo.aggregate([
  //       {
  //         $lookup: {
  //           from: "averagesemesterscores",
  //           localField: "studentId",
  //           foreignField: "studentId",
  //           as: "averageScores",
  //         },
  //       },
  //       {
  //         $lookup: {
  //           from: "trainingpoints",
  //           localField: "studentId",
  //           foreignField: "studentId",
  //           as: "trainingPoints",
  //         },
  //       },
  //       {
  //         $addFields: {
  //           averageScores: {
  //             $sortArray: {
  //               input: "$averageScores",
  //               sortBy: {
  //                 academicYear: 1,
  //                 semester: 1,
  //               },
  //             },
  //           },
  //           trainingPoints: {
  //             $sortArray: {
  //               input: "$trainingPoints",
  //               sortBy: {
  //                 academicYear: 1,
  //                 semester: 1,
  //               },
  //             },
  //           },
  //         },
  //       },
  //     ]);

  //     // Xử lý dữ liệu cho CSV
  //     const csvData = students.map((student) => {
  //       const averageScoresColumns = Array(22).fill(-1); // 22 cột cho averageScores
  //       const trainingPointsColumns = Array(16).fill(-1); // 16 cột cho trainingPoints

  //       // Điểm trung bình học kỳ
  //       student.averageScores.forEach((score) => {
  //         const semesterIndex =
  //           (parseInt(score.academicYear.split("-")[0]) - 2022) * 3 +
  //           (parseInt(score.semester) - 1);
  //         if (semesterIndex >= 0 && semesterIndex < 22) {
  //           averageScoresColumns[semesterIndex] =
  //             (score.averageScore || 0) / 10; // Chuyển về thang điểm 1
  //         }
  //       });

  //       // Điểm rèn luyện
  //       student.trainingPoints.forEach((point) => {
  //         const termIndex =
  //           (parseInt(point.academicYear.split("-")[0]) - 2022) * 2 +
  //           (parseInt(point.semester) - 1);
  //         if (termIndex >= 0 && termIndex < 16) {
  //           trainingPointsColumns[termIndex] = (point.trainingPoint || 0) / 100; // Chuyển về thang điểm 1
  //         }
  //       });

  //       // Xử lý các cột phương thức xét tuyển
  //       const admissionMethodColumns = {
  //         THPT:
  //           student.admissionMethod === "THPT" ||
  //           student.admissionMethod === "ƯT-ĐHQG" ||
  //           student.admissionMethod === "ƯT-Bộ"
  //             ? 1
  //             : 0,
  //         ĐGNL: student.admissionMethod === "ĐGNL" ? 1 : 0,
  //         OTHER: student.admissionMethod === "Khác" ? 1 : 0,
  //       };

  //       return {
  //         mssv: student.studentId, // Đổi tên cột studentId thành mssv
  //         diem_tt: student.admissionScore || "N/A", // Đổi tên cột admissionScore thành diem_tt
  //         OTHER: admissionMethodColumns.OTHER,
  //         THPT: admissionMethodColumns.THPT,
  //         ĐGNL: admissionMethodColumns["ĐGNL"],
  //         majorCode: student.majorCode || "N/A",
  //         educationSystem: student.educationSystem || "N/A",
  //         faculty: student.faculty || "N/A",
  //         placeOfBirth: student.placeOfBirth || "N/A",
  //         ...Object.fromEntries(
  //           averageScoresColumns.map((score, i) => [`sem${i + 1}`, score])
  //         ),
  //         ...Object.fromEntries(
  //           trainingPointsColumns.map((point, i) => [`term${i + 1}`, point])
  //         ),
  //       };
  //     });

  //     // Định nghĩa các cột CSV
  //     const fields = [
  //       "mssv", // Đổi tên cột studentId thành mssv
  //       "diem_tt", // Đổi tên cột admissionScore thành diem_tt
  //       "OTHER", // Cột cho phương thức xét tuyển Khác
  //       "THPT", // Cột cho phương thức xét tuyển THPT
  //       "ĐGNL", // Cột cho phương thức xét tuyển ĐGNL
  //       "majorCode",
  //       "educationSystem",
  //       "faculty",
  //       "placeOfBirth",
  //       ...Array.from({ length: 22 }, (_, i) => `sem${i + 1}`),
  //       ...Array.from({ length: 16 }, (_, i) => `term${i + 1}`),
  //     ];

  //     const json2csvParser = new Parser({ fields });
  //     const csv = json2csvParser.parse(csvData);

  //     // Lưu file CSV
  //     const folderPath = path.join(process.cwd(), "data_input");
  //     if (!fs.existsSync(folderPath)) {
  //       fs.mkdirSync(folderPath);
  //     }
  //     const filePath = path.join(folderPath, "students_data.csv");
  //     fs.writeFileSync(filePath, csv);

  //     const outputFolderPath = path.join(process.cwd(), "output");
  //     if (!fs.existsSync(outputFolderPath)) {
  //       fs.mkdirSync(outputFolderPath, { recursive: true }); // Tạo thư mục nếu chưa tồn tại
  //     }
  //     const timeout = 60000;

  //     const watcher = fs.watch(outputFolderPath, (eventType, filename) => {
  //       if (eventType === "change" && filename === "predicted_classes.txt") {
  //         try {
  //           const outputFilePath = path.join(outputFolderPath, filename);
  //           const fileContent = fs.readFileSync(outputFilePath, "utf8");
  //           const resultArray = fileContent
  //             .split("\n")
  //             .filter((line) => line.trim() !== "");

  //           const cleanedArray = resultArray.map((line) => line.trim());

  //           watcher.close();
  //           res.status(200).json({
  //             message: "File updated and processed successfully",
  //             data: cleanedArray,
  //           });
  //         } catch (err) {
  //           watcher.close();
  //           res.status(500).json({ message: "Error reading file", error: err });
  //         }
  //       }
  //     });

  //     // Dừng lắng nghe sau timeout
  //     setTimeout(() => {
  //       watcher.close();
  //       res.status(408).json({ message: "Timeout waiting for file update" });
  //     }, timeout);
  //   } catch (error) {
  //     res
  //       .status(500)
  //       .json({ message: "Error processing students' data", error });
  //   }
  // });

  app.get("/api/students-to-csv", async (req, res) => {
    try {
      const students = await StudentInfo.aggregate([
        {
          $lookup: {
            from: "averagesemesterscores",
            localField: "studentId",
            foreignField: "studentId",
            as: "averageScores",
          },
        },
        {
          $lookup: {
            from: "trainingpoints",
            localField: "studentId",
            foreignField: "studentId",
            as: "trainingPoints",
          },
        },
        {
          $addFields: {
            averageScores: {
              $sortArray: {
                input: "$averageScores",
                sortBy: {
                  academicYear: 1,
                  semester: 1,
                },
              },
            },
            trainingPoints: {
              $sortArray: {
                input: "$trainingPoints",
                sortBy: {
                  academicYear: 1,
                  semester: 1,
                },
              },
            },
          },
        },
      ]);
  
      // Xử lý dữ liệu cho CSV
      const csvData = students.map((student) => {
        const averageScoresColumns = Array(22).fill(-1);
        const trainingPointsColumns = Array(16).fill(-1);
  
        student.averageScores.forEach((score) => {
          const semesterIndex =
            (parseInt(score.academicYear.split("-")[0]) - 2022) * 3 +
            (parseInt(score.semester) - 1);
          if (semesterIndex >= 0 && semesterIndex < 22) {
            averageScoresColumns[semesterIndex] = (score.averageScore || 0) / 10;
          }
        });
  
        student.trainingPoints.forEach((point) => {
          const termIndex =
            (parseInt(point.academicYear.split("-")[0]) - 2022) * 2 +
            (parseInt(point.semester) - 1);
          if (termIndex >= 0 && termIndex < 16) {
            trainingPointsColumns[termIndex] = (point.trainingPoint || 0) / 100;
          }
        });
  
        const admissionMethodColumns = {
          THPT:
            student.admissionMethod === "THPT" ||
            student.admissionMethod === "ƯT-ĐHQG" ||
            student.admissionMethod === "ƯT-Bộ"
              ? 1
              : 0,
          ĐGNL: student.admissionMethod === "ĐGNL" ? 1 : 0,
          OTHER: student.admissionMethod === "Khác" ? 1 : 0,
        };
  
        return {
          mssv: student.studentId,
          diem_tt: student.admissionScore || "N/A",
          OTHER: admissionMethodColumns.OTHER,
          THPT: admissionMethodColumns.THPT,
          ĐGNL: admissionMethodColumns["ĐGNL"],
          majorCode: student.majorCode || "N/A",
          educationSystem: student.educationSystem || "N/A",
          faculty: student.faculty || "N/A",
          placeOfBirth: student.placeOfBirth || "N/A",
          ...Object.fromEntries(
            averageScoresColumns.map((score, i) => [`sem${i + 1}`, score])
          ),
          ...Object.fromEntries(
            trainingPointsColumns.map((point, i) => [`term${i + 1}`, point])
          ),
        };
      });
  
      const fields = [
        "mssv",
        "diem_tt",
        "OTHER",
        "THPT",
        "ĐGNL",
        "majorCode",
        "educationSystem",
        "faculty",
        "placeOfBirth",
        ...Array.from({ length: 22 }, (_, i) => `sem${i + 1}`),
        ...Array.from({ length: 16 }, (_, i) => `term${i + 1}`),
      ];
  
      const json2csvParser = new Parser({ fields });
      const csv = json2csvParser.parse(csvData);
  
      const folderPath = path.join(process.cwd(), "data_input");
      if (!fs.existsSync(folderPath)) fs.mkdirSync(folderPath);
  
      const filePath = path.join(folderPath, "students_data.csv");
      fs.writeFileSync(filePath, csv);
  
      const outputFolderPath = path.join(process.cwd(), "output");
      if (!fs.existsSync(outputFolderPath)) {
        fs.mkdirSync(outputFolderPath, { recursive: true });
      }
  
      const outputFilePath = path.join(outputFolderPath, "predicted_classes.txt");
  
      // Chờ file được cập nhật
      const timeout = 60000; // 60 giây
      const interval = 500; // Kiểm tra mỗi 500ms
      const startTime = Date.now();
  
      const checkFileUpdated = () =>
        new Promise((resolve, reject) => {
          const intervalId = setInterval(() => {
            if (fs.existsSync(outputFilePath)) {
              const stats = fs.statSync(outputFilePath);
              if (Date.now() - stats.mtimeMs < 5000) {
                clearInterval(intervalId);
                resolve(true);
              }
            }
            if (Date.now() - startTime > timeout) {
              clearInterval(intervalId);
              reject(new Error("Timeout waiting for file update"));
            }
          }, interval);
        });
  
      await checkFileUpdated();
  
      const fileContent = fs.readFileSync(outputFilePath, "utf8");
      const resultArray = fileContent.split("\n").filter((line) => line.trim() !== "");
  
      res.status(200).json({
        message: "File updated and processed successfully",
        data: resultArray.map((line) => line.trim()),
      });
    } catch (error) {
      res.status(500).json({ message: "Error processing students' data", error });
    }
  });
}

export default webInitRouterStudent;
