# CyberEther Examples Guide

Welcome to the examples directory, which provides insights into various methods of running CyberEther.

## Dynamic Mode
In the dynamic mode, CyberEther is run by manually creating modules through its graphical interface.

- **What's Not Here:** This examples folder doesn't showcase a demonstration for dynamic mode. Why? The `main.cc` script in the base directory serves as the most basic example.
  
- **How It Works:** The `main.cc` script loads the configuration from a YAML file and initializes the graph during program startup. 

- **Where to Find Configuration Examples:** Check out the `flowgraphs` directory for multiple YAML configuration samples.

- **Interface Elements:** When operating in dynamic mode, the interface elements are defined within the Bundle class. This is different from the static mode where they are defined using C++.

## Static Mode
Static mode is the traditional method of setting up a graph in CyberEther. In this mode, modules are manually interconnected using C++.

- **User Flexibility:** One of the key differences from dynamic mode is that users can design their own interface elements and logic to interact with the modules.

- **Reusing Bundle Interfaces:** Even in static mode, interfaces from Bundles can be incorporated.

#### Hello World
This example provides the essentials needed to describe a flowgraph in CyberEther.

#### Interface
This example delves deeper, illustrating an intermediate graph that includes hand-coded interactive interface elements.