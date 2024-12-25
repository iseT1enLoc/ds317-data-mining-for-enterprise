# DS317-data-mining-for-enterprise

# TTAUGG: Tabular and Time Series approach for UIT student graduation grade.

---

## 1. Project overview

---

This repository provides an in-depth understanding of the project and its core components. It includes a detailed introduction, outlining the purpose, scope, and expected outcomes. By reading through this document, you will gain insight into the project's objectives, the problems it addresses, and the solutions it proposes.

We want to give many thank to lecture **NGUYEN THI ANH THU** for giving us meaningful advices and instruction through out the course.

Team members:

| Student name | ID |
| --- | --- |
| Nguyen Vo Tien Loc | 22520792 |
| Truong Hoai Bao | 22520126 |
| Pham Van Duy | 22520341 |
| Nguyen Quang Dang | 22520191 |

## 2. Data description

- Data entries in our dataset cover the period the period between 2013 and 2019 of UIT student performance. Each feature's name and corresponding column name in the CSV file are explained in the following table.

| Column name in CSV file | Description |
| --- | --- |
| mssv | Student identity  |
| diem_tt | Admission score |
| OTHER | The value is equal to **1** if the student admission method is not in {THPT, ĐGNL} |
| THPT | 
The value is equal to **1** if the student **admission method** is THPT (High School Graduation Exam)

 |
| ĐGNL | The value is equal to **1** if the student **admission method** is   ĐGNL(**Competency Assessment Examination**) |
| chuyennganh2_7340122_CLC | The value is equal to **1** if the student studied  Ecommerce Major in high quality program. |
| chuyennganh2_7340122_CQ | The value is equal to **1** if the student studied ECommerce Major in general program. |
| chuyennganh2_7480101_CLC | The value is equal to **1** if the student studied Computer Science Major in high quality program. |
| chuyennganh2_7480101_CQ | The value is equal to **1** if the student studied Computer Science Major in general program. |
| chuyennganh2_7480101_CTTN | The value is equal to **1** if the student studied Computer Science Major in honored program. |
| chuyennganh2_7480102_CLC | The value is equal to **1** if the student studied Data Commnication network major in high-quality program. |
| chuyennganh2_7480102_CQ | The value is equal to **1** if the student studied Data Commnication network major in general program. |
| chuyennganh2_7480103_CLC | The value is equal to **1** if the student studied Software engineering major in high-quality program. |
| chuyennganh2_7480103_CQ | The value is equal to **1** if the student studied Software engineering major in general program. |
| chuyennganh2_7480104_CLC | The value is equal to **1** if the student studied Information System major in high-quality program. |
| chuyennganh2_7480104_CQ | The value is equal to **1** if the student studied Information System major in high program. |
| chuyennganh2_7480104_CTTT | The value is equal to **1** if the student studied Information System major in advanced program. |
| chuyennganh2_7480106_CLC | The value is equal to **1** if the student studied Computer engineering major in high-quality program. |
| chuyennganh2_7480106_CQ | Computer engineering major in general program. |
| chuyennganh2_7480109_CQ | Data science major in general program. |
| chuyennganh2_7480201_CLC | Information Technology major in high-quality program. |
| chuyennganh2_7480201_CQ | Information Technology major in general program. |
| chuyennganh2_7480202_CLC | Cyber security major in high-quality program. |
| chuyennganh2_7480202_CQ | Cyber security major in general program. |
| chuyennganh2_7480202_CTTN | Cyber security major in honored program. |
| hedt_CLC | The value is equal to **1** if the student’s class is in high quality education program. |
| hedt_CNTN | The value is equal to **1** if the student’s class is in honor program. |
| hedt_CQUI | The value is equal to **1** if the student’s class is in general education program. |
| hedt_CTTT | The value is equal to **1** if the student’s class is in advanced program. |
| hedt_KSTN | The value is equal to **1** if the student’s class is in honor engineer class. |
| khoa_CNPM | The value is equal to **1** if the faculty of the student is Software Engineering. |
| khoa_HTTT | The value is equal to **1** if the student studied in the faculty of Information System. |
| khoa_KHMT | The value is equal to **1** if the student studied in the faculty of Computer Science. |
| khoa_KTMT | The value is equal to **1** if the student studied in the faculty of Computer Engineering. |
| khoa_KTTT | The value is equal to **1** if the student studied in the faculty of Information Science and Engineering. |
| khoa_MMT&TT | The value is equal to **1** if the student studied in the faculty of Computer Networks and Communications. |
| noisinh_0 | Northern Mountainous Provinces(Các tỉnh miền núi phía Bắc) |
| noisinh_1 | Red River Delta Provinces(Các tỉnh đồng bằng sông Hồng) |
| noisinh_2 | Central Provinces(Các tỉnh miền Trung) |
| noisinh_3 | Central Highlands Provinces(Các tỉnh Tây Nguyên) |
| noisinh_4 | Southeastern Provinces(Các tỉnh Đông Nam Bộ) |
| noisinh_5 | Mekong Delta Provinces(Các tỉnh Đồng Bằng Mê Kong) |
| noisinh_6 | Foreign Countries(Sinh viên sinh ở nước ngoài) |
| sem1 | Cumulative GPA for the first semester. |
| sem2 | Cumulative GPA for the second semester. |
| sem3 | Cumulative GPA for the first-year summer semester. |
| sem4 | Cumulative GPA for the fourth semester. |
| sem5 | Cumulative GPA for the fifth semester. |
| sem6 | Cumulative GPA for the second-year summer semester. |
| sem7 | Cumulative GPA for the fifth semester. |
| sem8 | Cumulative GPA for the sixth semester. |
| sem9 | Cumulative GPA for the third-year summer semester. |
| sem10 | Cumulative GPA for the seventh semester. |
| sem11 | Cumulative GPA for the eighth semester. |
| sem12 | Cumulative GPA for the fourth-year summer semester. |
| sem13 | Cumulative GPA for the ninth semester. |
| sem14 | Cumulative GPA for the tenth semester. |
| sem15 | Cumulative GPA for the fifth-year summer semester. |
| sem16 | Cumulative GPA for the eleventh semester. |
| sem17 | Cumulative GPA for the twelfth semester |
| sem18 | Cumulative GPA for the sixth-year summer semester. |
| sem19 | Cumulative GPA for the thirteenth semester. |
| sem20 | Cumulative GPA for the fourteenth semester |
| sem21 | Cumulative GPA for the seventh-year summer semester. |
| sem22 | Cumulative GPA for the fifteenth semester. |
| term1 | Conduct grade mark for the first semester. |
| term2 | Conduct grade mark for the second semester. |
| term3 | Conduct grade mark for the third semester. |
| term4 | Conduct grade mark for the fourth semester. |
| term5 | Conduct grade mark for the fifth semester. |
| term6 | Conduct grade mark for the sixth semester. |
| term7 | Conduct grade mark for the seventh semester. |
| term8 | Conduct grade mark for the eighth semester. |
| term9 | Conduct grade mark for the ninth semester. |
| term10 | Conduct grade mark for the tenth semester. |
| term11 | Conduct grade mark for the eleventh semester. |
| term12 | Conduct grade mark for the twelfth semester. |
| term13 | Conduct grade mark for the thirteenth semester. |
| term14 | Conduct grade mark for the fourteenth semester. |
| term15 | Conduct grade mark for the fifteenth semester. |
| term16 | Conduct grade mark for the sixteenth semester. |
| label | The value is equal to 0, 1, 2, 3 corresponding to (Excellent, Very Good), Good, Ordinary, Not Completed |
- The above dataset contains 5181 entries including **1280, 2645, 933, 323** entries in (Excellent, Very Good), Good, Ordinary, Not Completed label respectively carefully extracted and verified from data preprocessing process from the raw dataset.

## 3. Our approach

The below image show the full process of addressing the problem:

![image/png](media/image.png)

## 4. Website

### Website Overview:

This project is a web-based application that integrates a predictive model for forecasting student graduation classification into a student data management system. The application is specifically designed to assist academic advisors in managing and analyzing student data for a specific class. By providing tools for data visualization and predictive analytics, the system enables advisors to monitor student performance and implement timely interventions for those at risk of poor outcomes.

### Key Features:

- **Student Information Management:** Manage comprehensive student information, including:
    - Student ID
    - Phone number
    - Email
    - Place of birth
    - Admission scores
    - Major
    - Academic department
    - Semester GPA (Grade Point Average)
    - Semester conduct scores
    - Other essential personal and academic details
- **Predictive Analytics:** Employ a trained model to predict graduation classifications based on historical and current data.
- **Business Intelligence (BI) Dashboard:** Provide professional BI tools for academic advisors to:
    - Visualize data through interactive charts and graphs.
    - Analyze trends in academic and conduct performance.
    - Identify students who require academic support.
- **Actionable Insights:** Help advisors design interventions based on predicted outcomes to improve student success rates.

## Application Interfaces

### 4.1. Login Interface

This screen provides a secure login system for authorized users. The interface includes:

- Fields for username and password input.
- A clean and intuitive design that ensures ease of use.

![image.png](media/image%201.png)

*A simple login page with a welcoming design, input fields for username and password, and a "Submit" button.*

### 4.2. Dashboard Interface

The dashboard provides a comprehensive overview of key data metrics. It includes:

- Interactive charts visualizing:
    - Average GPA trends over recent semesters.
    - Conduct scores for the latest semester.
    - Predicted academic performance classifications.

![media/image_2024-12-25_161001224.png](media/image_2024-12-25_161001224.png)

*A vibrant dashboard showing bar charts for GPA distribution, line graphs for conduct scores over time, and pie charts for predicted performance classifications. Dropdown menus and filters are available at the top for customization.*

### 4.3. Student List Interface

This interface displays a sortable and searchable table of student records, including:

- Basic student information:
    - Student ID
    - Name
    - Gender
    - Place oF birth
    - Phone number
    - Email
    - Status
- Pagination for ease of navigation.

![media/image_2024-12-25_161345621.png](media/image_2024-12-25_161345621.png)

*A tabular view with headers for Student ID, Name, Phone, and Email. Each row represents a student, with a search bar at the top for quick lookup.*

![media/image_2024-12-25_162002331.png](media/image_2024-12-25_162002331.png)

*This interface extends the basic student list by including a "Status" column, which displays predicted graduation classifications, predicted outcomes based on the trained model.*

### 4.5. Detailed Student Information Interface

This screen provides in-depth details about an individual student, including:

- Personal and academic information.
- Interactive charts visualizing:
    - GPA trends.
    - Conduct score variations.

![media/image_2024-12-25_164447473.png](media/image_2024-12-25_164447473.png)

![media/image_2024-12-25_161542599.png](media/image_2024-12-25_161542599.png)

*A profile page with sections for personal information, followed by a line chart of semester GPAs and a bar chart for conduct scores. The design is clean, with tabs for navigating different sections of the student’s profile.*

![media/image_2024-12-25_161702440.png](media/image_2024-12-25_161702440.png)

*Similar to the detailed profile interface but includes a highlighted section showing the predicted graduation classification. A pie chart or bar graph illustrates the key contributing factors.*

## Technologies Used

- **Frontend:** React.ts for a dynamic and responsive user interface.
- **Backend:** Node.js with Express for API management.
- **Database:** MongoDB for secure and structured data storage.
- **Machine Learning:** A predictive model trained using Python and integrated with the web application.
- **Visualization:** MUI  x Chart for interactive data visualization.

## 5. Setup instruction

### Backend Setup

1. Navigate to the `backend` directory.
    
    ```jsx
    cd .\web\academic-advisor-be
    ```
    
2. Install dependencies:
    
    ```
    npm install
    ```
    
3. Set up the MongoDB database:
    - Ensure MongoDB is running locally or provide a remote connection string in the `.env` file.
    - Example `.env` configuration:
        
        Connect us to get URI mongodb (dangnguyenquangit@gmail.com)
        
4. Start the backend server:
    
    ```
    npm start
    ```
    

### Frontend Setup

1. Navigate to the `frontend` directory.
    
    ```jsx
    cd .\web\academic-advisor-fe
    ```
    
2. Install dependencies:
    
    ```
    npm install
    ```
    
3. Start the development server:
    
    ```
    npm run dev
    ```
    
4. Access the frontend at `http://localhost:5173`.

### Model Setup

1. Navigate to the `model` directory.

```jsx
cd .\web\academic-advisor-be\python
```

1. Create a virtual environment:
    
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
    
2. Install dependencies:
    
    ```
    pip install pandas matplotlib watchdog torch torchmetrics pytorch-lightning
    ```
    
3. Run the `processing.py` file to execute the model:

```jsx
python processing.py
```

## 6. References

[(PDF) Aggregating Time Series and Tabular Data in Deep Learning Model for University Students’ GPA Prediction](https://www.researchgate.net/publication/352343704_Aggregating_Time_Series_and_Tabular_Data_in_Deep_Learning_Model_for_University_Students%27_GPA_Prediction)