import classnames from 'classnames';
import React from 'react';
import Input from './components/Input';
import { EditTextDefaultProps, EditTextPropTypes } from './propTypes';
import styles from './styles.module.css';

export default function EditText({
  id,
  name,
  className,
  placeholder,
  inline,
  style,
  readonly,
  type,
  value,
  defaultValue,
  formatDisplayText,
  onEditMode,
  onChange,
  onSave,
  onBlur,
  showEditButton,
  editButtonContent,
  editButtonProps
}) {
  const inputRef = React.useRef(null);
  const [previousValue, setPreviousValue] = React.useState('');
  const [savedText, setSavedText] = React.useState('');
  const [editMode, setEditMode] = React.useState(false);

  React.useEffect(() => {
    if (defaultValue !== undefined) {
      setPreviousValue(defaultValue);
      setSavedText(defaultValue);
    }
  }, [defaultValue]);

  React.useEffect(() => {
    if (value !== undefined) {
      setSavedText(value);
      if (!editMode) {
        setPreviousValue(value);
      }
    }
  }, [value, editMode]);

  const handleClickDisplay = () => {
    if (readonly || showEditButton) return;
    setEditMode(true);
    onEditMode();
  };

  const handleClickEditButton = () => {
    setEditMode(true);
    onEditMode();
  };

  const handleBlur = (save = true) => {
    if (inputRef.current) {
      const { name: inputName, value: inputValue } = inputRef.current;
      if (save && previousValue !== inputValue) {
        onSave({
          name: inputName,
          value: inputValue,
          previousValue: previousValue
        });
        setSavedText(inputValue);
        setPreviousValue(inputValue);
      } else if (!save) {
        onChange(previousValue);
      }
      setEditMode(false);
      onBlur();
    }
  };

  const handleKeydown = (e) => {
    if (e.keyCode === 13 || e.charCode === 13) {
      handleBlur();
    } else if (e.keyCode === 27 || e.charCode === 27) {
      handleBlur(false);
    }
  };

  const handleFocus = (e) => {
    if (type === 'text') {
      e.currentTarget.setSelectionRange(
        e.currentTarget.value.length,
        e.currentTarget.value.length
      );
    }
  };

  const renderDisplayMode = () => {
    return (
      <div
        className={classnames(styles.displayContainer, {
          [styles.inline]: inline
        })}
      >
        <div
          id={id}
          className={classnames(
            styles.label,
            styles.shared,
            {
              [styles.placeholder]: placeholder && !savedText,
              [styles.inline]: inline,
              [styles.readonly]: readonly || showEditButton
            },
            className
          )}
          onClick={handleClickDisplay}
          style={style}
        >
          {formatDisplayText(savedText) || placeholder}
        </div>
        {showEditButton && !readonly && (
          <button
            type='button'
            className={styles.editButton}
            {...editButtonProps}
            onClick={handleClickEditButton}
          >
            {editButtonContent}
          </button>
        )}
      </div>
    );
  };

  const renderEditMode = (controlled) => {
    const sharedProps = {
      inputRef: inputRef,
      handleBlur: handleBlur,
      handleKeydown: handleKeydown,
      handleFocus: handleFocus,
      props: { id, inline, className, style, type, name }
    };
    return controlled ? (
      <Input
        {...sharedProps}
        value={value}
        onChange={(e) => {
          onChange(e.target.value);
        }}
      />
    ) : (
      <Input {...sharedProps} defaultValue={savedText} />
    );
  };

  return !readonly && editMode
    ? renderEditMode(value !== undefined)
    : renderDisplayMode();
}

EditText.defaultProps = EditTextDefaultProps;
EditText.propTypes = EditTextPropTypes;