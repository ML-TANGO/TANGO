
const projectIdReducer = (state = "", action) => {
  switch (action.type)
  {
    case "change_project_id":
      return (state = action.id);
    default:
      return state;
  }
};

export default projectIdReducer;
