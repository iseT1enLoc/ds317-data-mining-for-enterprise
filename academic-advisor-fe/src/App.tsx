//import { useState } from "react";
//import reactLogo from "./assets/react.svg";
//import viteLogo from "/vite.svg";
//import "./App.css";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { Fragment } from "react";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import {publicRoutes, privateRoutes} from "./routes";
import DefaultLayout from "./components/layout/Default-Layout";
function App(): JSX.Element {
  return (
    <Router>
      <div className="App">
        <Routes>
          {/* Public Routes */}
          {publicRoutes.map((route, index) => {
            const Layout = route.layout === null ? Fragment : DefaultLayout;
            const Page = route.component;
            return (
              <Route
                key={index}
                path={route.path}
                element={
                  <Layout>
                    <Page />
                  </Layout>
                }
              />
            );
          })}
           {/* Private Routes */}
           {privateRoutes.map((route, index) => {
            const Layout = route.layout === null ? Fragment : DefaultLayout;
            const Page = route.component;
            return (
              <Route
                key={index}
                path={route.path}
                element={
                  <Layout>
                    <Page />
                  </Layout>
                }
              />
            );
          })}
        </Routes>
        <ToastContainer />
      </div>
    </Router>
  );
}

export default App;
