import React from "react"
import PropTypes from "prop-types"
import styled from "styled-components"
import { Card, CardImg, CardText, CardBody, CardTitle, Col, Row } from "reactstrap"

const Wrapper = styled.div`
  display: inline-block;
  width: 330px;
  height: 250px;
  border: 0.5px #4360fb dashed;
  padding: 0.5rem;
  margin: 1rem;
`

const StyledCard = styled(Card)`
  padding-bottom: 0px;
  &:hover {
    cursor: pointer;
  }
`

const StyledCarImg = styled(CardImg)`
  width: 100%;
  height: 100%;
  padding: 0.2rem;
`

const SyteldCardTitle = styled(CardTitle)`
  color: white !important;
  font-size: 20px;
`

function DataSetCard(props) {
  return (
    <Wrapper>
      <StyledCard onClick={props._handleClick}>
        <Row noGutters className="h-100">
          <Col xl={7}>
            <StyledCarImg src={props.src}></StyledCarImg>
          </Col>
          <Col xl={5}>
            <CardBody>
              <SyteldCardTitle>{props.title}</SyteldCardTitle>
              <CardText>{props.body}</CardText>
            </CardBody>
          </Col>
        </Row>
      </StyledCard>
    </Wrapper>
  )
}

DataSetCard.propTypes = {
  _handleClick: PropTypes.func,
  src: PropTypes.string,
  title: PropTypes.string,
  body: PropTypes.string
}

export default DataSetCard
