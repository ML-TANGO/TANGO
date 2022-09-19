import classnames from 'classnames';
import React from 'react';
import styles from '../styles.module.css';

const Textarea = ({
  props: { id, rows, className, style, name },
  inputRef,
  handleBlur,
  handleKeydown,
  ...rest
}) => {
  return (
    <textarea
      id={id}
      className={classnames(styles.shared, className)}
      style={style}
      ref={inputRef}
      rows={rows}
      name={name}
      onBlur={handleBlur}
      onKeyDown={handleKeydown}
      autoFocus
      onFocus={(e) =>
        e.currentTarget.setSelectionRange(
          e.currentTarget.value.length,
          e.currentTarget.value.length
        )
      }
      {...rest}
    />
  );
};

export default Textarea;