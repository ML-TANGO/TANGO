
const projectNameReducer = (state = "", action) => {
  switch (action.type)
  {
    case "change_project_name":
      return (state = action.name);
    default:
      return state;
  }
};

export default projectNameReducer;
