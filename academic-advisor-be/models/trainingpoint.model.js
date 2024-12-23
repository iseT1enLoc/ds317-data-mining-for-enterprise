import { Schema, model } from "mongoose";

const trainingPointSchema = new Schema({
  studentId: {
    type: String,
    required: true,
  },
  trainingScore: {
    type: Number,
    required: true,
    min: 0,
    max: 100,
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

export const TrainingPoint = model("TrainingPoint", trainingPointSchema);
