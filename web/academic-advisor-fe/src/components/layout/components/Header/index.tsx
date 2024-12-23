import { FaRegBell } from "react-icons/fa";
import { IoMailOutline } from "react-icons/io5";
import { MdLogout } from "react-icons/md";
function Header() {
  return (
    <div className="h-16 w-full bg-white flex">

      <div className="flex items-center justify-end w-full gap-4">
        <IoMailOutline size={22} className="cursor-pointer hover:text-red-400" />
        <FaRegBell size={22} className="cursor-pointer hover:text-red-400" />
        <button className="inline-flex items-center gap-2 px-4 hover:text-red-400">
          <span>Log out</span>
          <MdLogout size={22} />
        </button>
      </div>
    </div>
  );
}

export default Header;
