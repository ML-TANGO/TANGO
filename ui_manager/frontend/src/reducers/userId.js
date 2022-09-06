
const userIdReducer = (state = "", action) => {
  switch (action.type)
  {
    case "set_user_id":
      return (state = action.name);
    default:
      return state;
  }
};

export default userIdReducer;
