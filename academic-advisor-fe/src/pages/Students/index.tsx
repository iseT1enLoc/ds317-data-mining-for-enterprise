import MUIDataTable from "mui-datatables";

const columns = ["Student ID", "Name", "Class", "Major", "GPA", "Action", "Status", ];
function Students(): JSX.Element {
  const data = [
    ["Joe James", "Test Corp", "Yonkers", "NY"],
    ["John Walsh", "Test Corp", "Hartford", "CT"],
    ["Bob Herm", "Test Corp", "Tampa", "FL"],
    ["James Houston", "Test Corp", "Dallas", "TX"],
  ];
  const options = {
    filterType: "checkbox",
  };
  return (
    <div className="mt-10 mx-auto pb-4 min-h-screen">
      <div className="flex justify-center">
        <MUIDataTable
          title={"Employee List"}
          data={data}
          columns={columns}
          options={options}
        />
      </div>
    </div>
  );
}

export default Students;
