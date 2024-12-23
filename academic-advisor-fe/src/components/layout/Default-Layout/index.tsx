interface DefaultLayoutProps {
  children: React.ReactNode;
}
import Header from "../components/Header";
import Sidebar from "../components/Sidebar";
function DefaultLayout({ children }: DefaultLayoutProps) {
  return (
    <div className="dashboard w-screen-xl flex min-h-screen">
      <Sidebar />
      <div className="flex flex-col basis-4/5">
        <Header />
        <hr className="border w-full border-black/20" />
        <div className="flex bg-slate-100">{children}</div>
      </div>
    </div>
  );
}

export default DefaultLayout;
