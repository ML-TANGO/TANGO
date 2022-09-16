import React from "react"
import { useForm } from "react-hook-form"
import PropTypes from "prop-types"
import { InputGroup, InputGroupAddon, InputGroupText, Input } from "reactstrap"
import { MdSearch } from "react-icons/md"

function SearchBar(props) {
  const { register, handleSubmit } = useForm()

  return (
    <form className=" d-flex justify-content-center" onSubmit={handleSubmit(props._handleSubmit)}>
      <InputGroup>
        <Input className="form-control" name="search" type="text" placeholder="Search" aria-label="Search" innerRef={register} />
        <InputGroupAddon addonType="append">
          <InputGroupText>
            <MdSearch aria-hidden="true" />
          </InputGroupText>
        </InputGroupAddon>
      </InputGroup>
    </form>
  )
}

SearchBar.propTypes = {
  _handleSubmit: PropTypes.func,
  src: PropTypes.string,
  title: PropTypes.string,
  body: PropTypes.string
}

export default SearchBar
