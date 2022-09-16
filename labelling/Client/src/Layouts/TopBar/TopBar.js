import React from "react"
import { NavLink, Link } from "react-router-dom"
import { Navbar, Nav, NavItem } from "reactstrap"
import styled from "styled-components"
import { RiSettings4Line, RiDownload2Line } from "react-icons/ri"

const StyledNavItem = styled(NavItem)`
  margin-right: 40px;
`

const Logo = () => {
  return (
    <div
      style={{
        width: "50px",
        marginRight: "50px",
        marginLeft: "10px"
      }}
    >
      <div className="logo"></div>
    </div>
  )
}

const TopBar = () => {
  return (
    <div className="topbar" style={{ zIndex: 99 }}>
      <Navbar color="primary" className="navbar-dark bg-dark" expand="lg" style={{ flexWrap: "nowrap" }}>
        <div className="navbar-dark bg-dark navbar navbar-expand-lg bg-primary">
          <Link to={"/dashboard"}>
            <Logo />
          </Link>
        </div>
        <Nav className="mr-auto" navbar style={{ flexDirection: "row" }}>
          <StyledNavItem>
            <h4>
              <NavLink to={"/datasets"}>DataSets</NavLink>
            </h4>
          </StyledNavItem>
          <StyledNavItem>
            <h4>
              <NavLink to={"/aimodels"}>AI Models</NavLink>
            </h4>
          </StyledNavItem>
          <StyledNavItem>
            <h4>
              <NavLink to={"/inputSource"}>Sources</NavLink>
            </h4>
          </StyledNavItem>
          <StyledNavItem>
            <h4>
              <NavLink to={"/project"}>Projects</NavLink>
            </h4>
          </StyledNavItem>
          {/* <StyledNavItem>
            <h4>
              <NavLink to={"/monitor"}>Monitor</NavLink>
            </h4>
          </StyledNavItem> */}
        </Nav>
        {/* <NavbarText>
          <InputGroup>
            <InputGroupAddon addonType="prepend">
              <div className="topbar-search-icon">
                <MdSearch className="topbar-icon" style={{ fontSize: "25px", paddingTop: "0.1rem" }} />
              </div>
            </InputGroupAddon>
            <Input type="text" placeholder={"Search"} />
          </InputGroup>
        </NavbarText> */}
        <Link to="/downloadPretrain">
          <span>
            <RiDownload2Line className="ml-2 icon-pointer" style={{ fontSize: "25px" }} />
          </span>
        </Link>
        {process.env.BUILD === "EE" && (
          <Link to="/setting">
            <RiSettings4Line className="ml-2 icon-pointer" style={{ fontSize: "25px" }} />
          </Link>
        )}
      </Navbar>
    </div>
  )
}

export default TopBar
