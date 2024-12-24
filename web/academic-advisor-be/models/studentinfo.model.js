import { Schema, model } from "mongoose";

const studentInfoSchema = new Schema({
    studentId: {
        type: String,
        required: true,
        unique: true,
    },
    name: {
        type: String,
        required: true,
    },
    admissionMethod: {
        type: String,
        required: true,
    },
    gender: {
        type: String,
        required: true,
        enum: ["Male", "Female"],
    },
    faculty: {
        type: String,
        required: true,
    },
    educationSystem: {
        type: String,
        required: true,
    },
    admissionScore: {
        type: Number,
        required: true,
        min: 0,
    },
    placeOfBirth: {
        type: String,
        required: true,
    },
    phoneNumber: {
        type: String,
        required: true,
        match: /^\d{10,11}$/,
    },
    email: {
        type: String,
        required: true,
        match: /.+\@.+\..+/,
    },
});

export const StudentInfo = model("StudentInfo", studentInfoSchema);