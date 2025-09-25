import PropTypes from 'prop-types';
import React from 'react';
import EditIcon from './components/EditIcon';

const sharedDefaultProps = {
  id: undefined,
  name: undefined,
  nindex: undefined,
  className: undefined,
  value: undefined,
  formatDisplayText: (x) => x,
  defaultValue: undefined,
  placeholder: '',
  onSave: () => {},
  onChange: () => {},
  onEditMode: () => {},
  onBlur: () => {},
  style: {},
  readonly: false
};

const sharedPropTypes = {
  id: PropTypes.string,
  name: PropTypes.string,
  className: PropTypes.string,
  nindex: PropTypes.integer,
  value: PropTypes.string,
  formatDisplayText: PropTypes.func,
  defaultValue: PropTypes.string,
  placeholder: PropTypes.string,
  onSave: PropTypes.func,
  onChange: PropTypes.func,
  onEditMode: PropTypes.func,
  onBlur: PropTypes.func,
  style: PropTypes.oneOfType([PropTypes.object, PropTypes.array]),
  readonly: PropTypes.bool
};

export const EditTextPropTypes = {
  ...sharedPropTypes,
  type: PropTypes.string,
  inline: PropTypes.bool,
  showEditButton: PropTypes.bool,
  editButtonContent: PropTypes.any,
  editButtonProps: PropTypes.object
};

export const EditTextDefaultProps = {
  ...sharedDefaultProps,
  type: 'text',
  inline: false,
  showEditButton: false,
  editButtonContent: <EditIcon />,
  editButtonProps: {}
};

export const EditTextareaPropTypes = {
  ...sharedPropTypes,
  rows: PropTypes.number
};

export const EditTextareaDefaultProps = {
  ...sharedDefaultProps,
  rows: 3
};