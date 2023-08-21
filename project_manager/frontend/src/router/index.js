import Vue from "vue";
import VueRouter from "vue-router";
import ProjectDashboard from "@/pages/ProjectDashboard.vue";
import ProjectDetail from "@/pages/ProjectDetail.vue";
import LoginPage from "@/pages/LoginPage.vue";
import CreatAccountPage from "@/pages/CreateAccountPage.vue";
import NotFoundPage from "@/pages/NotFoundPage.vue";
import VisualizationPage from "@/pages/VisualizationPage.vue";
import DataManagement from "@/pages/DataManagement.vue";
import TargetManagement from "@/pages/TargetManagement.vue";

import Cookies from "universal-cookie";

Vue.use(VueRouter);

const routes = [
  {
    path: "/",
    redirect: "/project"
  },
  {
    path: "/login",
    component: LoginPage,
    meta: { permision: "guest" }
  },
  {
    path: "/create-account",
    component: CreatAccountPage,
    meta: { permision: "guest" }
  },
  {
    path: "/project",
    name: "projects",
    component: ProjectDashboard
  },
  {
    path: "/project/:id",
    name: "ProjectDetail",
    component: ProjectDetail
  },
  {
    path: "/target",
    name: "targets",
    component: TargetManagement
  },
  {
    path: "/data",
    name: "datasets",
    component: DataManagement
  },
  {
    path: "/visualization",
    name: "Visualization",
    component: VisualizationPage
  },
  {
    path: "*",
    name: "Not Found",
    component: NotFoundPage
  }
];

const router = new VueRouter({
  mode: "history",
  base: process.env.BASE_URL,
  routes
});

// /* 웹 브라우저 쿠키 정보 유무 확인 */
const isToken = () => new Cookies().get("TANGO_TOKEN");
const isUser = () => new Cookies().get("userinfo");

router.beforeEach(async (to, from, next) => {
  const { permision } = to.meta;

  if (permision === undefined)
    if (!!isToken() === false || !!isUser() === false) {
      next("/login");
    } else {
      next();
    }
  else {
    if (!!isToken() === false || !!isUser() === false) {
      next();
    } else {
      next("/");
    }
  }
});

export default router;
