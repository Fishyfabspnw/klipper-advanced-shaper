# Security Policy

## Supported versions

Only the latest published alpha is supported while the project is pre-1.0.

## Reporting a vulnerability

Do not open a public issue containing credentials, private printer data, network
addresses, configuration files, or unreleased exploit details. Use GitHub's
private security-advisory flow for this repository. Include the affected
version, reproduction steps, and impact, with secrets and captures redacted.

## Credential and printer policy

- The plugin never needs a printer's SSH credentials at runtime.
- Keep SSH authentication in an OS credential manager or an interactive prompt.
- Never place credentials in repository files, Git history, CI configuration,
  issues, pull requests, logs, test fixtures, or generated reports.
- Use key-based authentication with a passphrase where possible and rotate any
  credential accidentally disclosed.
- Raw captures and reports are private by default and ignored by Git.

The public-tree check is defense in depth, not a substitute for reviewing every
diff before it is published.
