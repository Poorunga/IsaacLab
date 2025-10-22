import argparse
from lxml import etree


# Function to add missing inertial tags, ensure unique collider paths, and update the robot name
def fix_urdf(urdf_path, robot_name, output_path=None):
    # Parse the URDF file
    urdf_tree = etree.parse(urdf_path)

    # Update the robot's name
    robot_tag = urdf_tree.xpath("//robot")
    if robot_tag:
        robot_tag[0].set("name", robot_name)
    else:
        print("Error: <robot> tag is missing in the URDF.")
        return

    # Add missing inertial elements
    for link in urdf_tree.xpath("//link"):
        if not link.xpath(".//inertial"):
            inertial = etree.Element("inertial")
            mass = etree.SubElement(inertial, "mass")
            mass.text = "1.0"  # Default mass value
            inertia = etree.SubElement(inertial, "inertia")
            inertia.set("ixx", "0.1")  # Default inertia values
            inertia.set("iyy", "0.1")
            inertia.set("izz", "0.1")
            link.append(inertial)

    # Ensure unique collider names
    collider_counter = 1
    for collision in urdf_tree.xpath("//collision"):
        collision_name = f"collider_{collider_counter}"
        collision.set("name", collision_name)
        collider_counter += 1

    # If no output path is provided, set it to input path with '_fixed' suffix
    if output_path is None:
        output_path = robot_name + ".urdf"

    # Save the fixed URDF to the output path
    urdf_tree.write(output_path, pretty_print=True)
    print(f"Fixed URDF has been saved to: {output_path}")


# Setup argument parser
def main():
    parser = argparse.ArgumentParser(
        description="Fixed URDF downloaded from Sapien(https://sapien.ucsd.edu/browse) to be compatible with IsaacSim URDF importer"
    )

    # Input path (optional), default is 'mobility.urdf'
    parser.add_argument(
        "--input",
        type=str,
        default="mobility.urdf",
        help="Path to the input URDF file (default is 'mobility.urdf')",
    )

    # Output path (optional)
    parser.add_argument(
        "--out",
        type=str,
        help="Path to save the fixed URDF file (default is input path with '_fixed' suffix)",
    )

    # Required robot name
    parser.add_argument(
        "--name", type=str, required=True, help="The name to assign to the <robot> tag"
    )

    args = parser.parse_args()

    # Call the fix function with provided arguments
    fix_urdf(args.input, args.name, args.out)


if __name__ == "__main__":
    main()
