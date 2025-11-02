# FICO Xpress Community projects and examples
This repository welcomes contributions from the broader FICO&reg; Xpress community. Submissions may include projects, examples, or integrations involving any component of the Xpress ecosystem, such as Xpress Solver, Xpress Insight, or the various APIs provided by Xpress (including Python, Mosel, Java, C++, and .NET). Contributions that demonstrate practical applications, innovative modeling techniques, or integration with open-source systems are particularly encouraged.

## Contents 
Project name | Description | Contributor | Date
-------------|-------------|-------------|-----|
[Bike Rebalancing Problem](BikeRebalancingProblem) | Rebalancing London's Bike-Sharing System: A FICO Xpress C++ API Case Study | Marco Deken | August 2024
[Vessel Schedule Recovery Problem](VesselSchedulingProblem) | An extended optimization model for the vessel schedule recovery problem | Jingwen He | August 2025
[Soccer Camp Location Problem](SoccerCampLocation) | Locating Soccer Camps at potential parks over a season | Carlos A. Zetina | November 2025

***Find detailed information about each project in the README file of the corresponding folder**.

## Contribution Guidelines

Thank you for your interest in contributing to this repository! We welcome contributions from the FICO Xpress community that showcase practical applications, tutorials or enhancements using:

- **Xpress Solver APIs** (e.g., Python, Mosel, Java, C++, .NET)
- **Xpress Insight** (Python or Mosel)

Your contributions can include:

- **Project examples** demonstrating optimization models or real-world use cases
- **Tutorials or notebooks** explaining concepts or workflows
- **Bug fixes or improvements** to existing examples or utilities
- **Integrations** with open-source tools, platforms, or data sources

### How to Contribute

1. **Fork the repository** to your GitHub account:
   - Go to the [GitHub page](https://github.com/fico-xpress/xpress-community) and click “Fork” in the top-right corner to create your own copy

2. **Clone your forked repository** to your local machine:
   ```bash
   git clone https://github.com/<your-username>/xpress-community.git
   cd xpress-community
   ```

3. **Create a new feature branch** for your contribution:  
   ```bash
   git checkout -b feature/my-new-example
   ```
   
4. **Add your contribution** in a clearly organized and properly named *subfolder*. Please include:
   - A brief `README.md` explaining the purpose, setup, and usage
   - Any required input data or configuration files
   - Comments or documentation in code where appropriate
   - **Ensure compatibility** with the latest version of Xpress, and **please submit models than can be executed with the Xpress community license** whenever possible

5. Stage and commit your changes:
   ```bash
   git add .
   git commit -m "Add example: [short description of your project]"
   ```

6. **Push your branch** to your fork:
   ```bash
   git push origin feature/my-new-example
   ```

7. **Submit a pull request**: Go to your fork on GitHub, you’ll see a prompt to open a pull request. Click *Compare & pull request*, then:
   - Set the base repository to `fico-xpress/xpress-community`
   - Set the base branch to `main`
   - Please include a clear and descriptive summary of your contribution and click *Create pull request*

All contributions to the FICO Xpress Community repository are subject to review by FICO staff to ensure quality, relevance, and compliance with community standards.

## Legal

See source code files for copyright notices.

## License

The examples in this repository are licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text. Some examples use FICO&reg; Xpress software. By running it, you agree to the Community License terms of the [Xpress Shrinkwrap License Agreement](https://www.fico.com/en/shrinkwrap-license-agreement-fico-xpress-optimization-suite-on-premises) with respect to the FICO&reg; Xpress software. See the [licensing options](https://www.fico.com/en/fico-xpress-trial-and-licensing-options) overview for additional details and information about obtaining a paid license.