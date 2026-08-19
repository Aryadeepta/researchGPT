# Security policy

Please do not report credentials or other sensitive findings in a public issue.
Use GitHub's private security-advisory reporting for this repository when
available, or contact the repository owner privately. Include the affected path
or workflow and enough redacted context to reproduce the issue; never include a
live credential in the report.

Credentials, `.env` files, model weights, research-run outputs, and generated
artifacts must not be committed. Before a public release, run
`scripts/public-release-audit` and resolve every failure or blocked remote
audit.
