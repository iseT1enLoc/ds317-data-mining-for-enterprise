import Dashboard from "../pages/Dashboard";
import Students from "~/pages/Students";
import Login from "~/pages/Login";
export interface Route {
  path: string;
  component: React.ComponentType;
  layout?: React.ComponentType | null;
}

const publicRoutes: Route[] = [
  { path: "/", component: Dashboard },
  { path: "/students", component: Students },
  
];

const privateRoutes: Route[] = [
  { path: "/login", component: Login, layout: null },
]

export {publicRoutes, privateRoutes}
