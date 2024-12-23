import { MdOutlineDashboard } from "react-icons/md";
import { PiStudentBold } from "react-icons/pi";
//import { IoSettingsOutline } from "react-icons/io5";
import { NavLink, useLocation } from "react-router-dom";
import UIT_LOGO from "~/assets/uit_logo.png";

function Sidebar(): JSX.Element {
  const { pathname } = useLocation();
  return (
    <div className="bg-[#152259] basis-1/5">
      <div className="UIT-Logo">
        <div className=" bg-[#152259] w-[252.67px] h-40 flex flex-col items-center justify-center gap-4">
          <div className="size-16 rounded-full bg-white flex items-center justify-center">
            <img
              src={UIT_LOGO}
              alt="Trường Đại Học Công Nghệ Thông Tin - ĐHQGHCM"
              className="w-14 h-auto "
            />
          </div>
          <p className="text-white text-sm">
            University of Information Technology
          </p>
        </div>
        <hr className=" border w-full border-white" />
      </div>
      <div className="sidebar mx-6 my-4">
        <ul className="text-white">
          <NavLink to="/">
            <li
              className={`flex items-center gap-5 py-2 my-1 hover:menu-active ${
                pathname === "/" ? "menu-active" : ""
              }`}
            >
              <MdOutlineDashboard size={24} className="ml-3" />
              <span className="ml-3">Dashboard</span>
            </li>
          </NavLink>
          <NavLink
            to="/students"
            className={({ isActive }) => (isActive ? "menu-active" : "")}
          >
            <li
              className={`flex items-center gap-5 py-2 my-1 hover:menu-active ${
                pathname === "/students" ? "menu-active" : ""
              }`}
            >
              <PiStudentBold size={24} className="ml-3" />
              <span className="ml-3">Students</span>
            </li>
          </NavLink>
        </ul>
      </div>
    </div>
  );
}

export default Sidebar;
