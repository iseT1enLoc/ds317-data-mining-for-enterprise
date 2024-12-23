import { Schema, model } from "mongoose";

const averageSemesterScoreSchema = new Schema({
  studentId: {
    type: String,
    required: true,
  },
  averageScore: {
    type: Number,
    required: true,
    min: 0,
    max: 10,
  },
  semester: {
    type: String,
    required: true,
  },
  academicYear: {
    type: String,
    required: true,
  },
});

export const AverageSemesterScore = model(
  "AverageSemesterScore",
  averageSemesterScoreSchema
);

