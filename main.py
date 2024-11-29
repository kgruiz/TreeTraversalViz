import random
from collections import deque

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import animation
from tqdm import tqdm


class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []

    def addChildren(self, children):
        self.children.extend(children)


def CreateRandomTree(numberOfNodes, maxChildren):
    if numberOfNodes <= 0:
        raise ValueError("Number of nodes must be positive.")
    if maxChildren < 1 and numberOfNodes > 1:
        raise ValueError(
            "Maximum number of children must be at least 1 for trees with more than one node."
        )

    root = TreeNode("A")
    nodes = [root]
    potentialParents = [root]
    currentLabel = 66  # ASCII for 'B'

    while len(nodes) < numberOfNodes and potentialParents:
        parent = random.choice(potentialParents)
        availableSlots = maxChildren - len(parent.children)
        remainingNodes = numberOfNodes - len(nodes)
        if availableSlots <= 0:
            potentialParents.remove(parent)
            continue
        numChildren = random.randint(1, min(availableSlots, remainingNodes))
        children = []
        for _ in range(numChildren):
            if len(nodes) >= numberOfNodes:
                break
            child = TreeNode(chr(currentLabel))
            currentLabel += 1
            children.append(child)
            nodes.append(child)
            potentialParents.append(child)
        parent.addChildren(children)
        if len(parent.children) >= maxChildren:
            potentialParents.remove(parent)

    while len(nodes) < numberOfNodes:
        parent = random.choice(nodes)
        if len(parent.children) < maxChildren:
            child = TreeNode(chr(currentLabel))
            currentLabel += 1
            parent.addChildren([child])
            nodes.append(child)
        else:
            if all(len(n.children) >= maxChildren for n in nodes):
                raise ValueError(
                    "Unable to assign all nodes within the maxChildren constraint."
                )

    return root


def BuildGraph(root):
    G = nx.DiGraph()

    def AddEdges(node):
        for child in node.children:
            G.add_edge(node.value, child.value)
            AddEdges(child)

    AddEdges(root)
    return G


def HierarchyPos(G, root, width=1.5, vertGap=0.5, vertLoc=0, xCenter=0.5):
    def _HierarchyPos(G, root, leftmost, width, vertGap, vertLoc, pos, parent=None):
        children = list(G.successors(root))
        # Remove or comment out the random shuffling to maintain order
        # random.shuffle(children)
        # Sort children based on their ASCII values
        children.sort(key=lambda child: child)
        if not children:
            pos[root] = (leftmost + width / 2, vertLoc)
            return leftmost + width, pos
        else:
            nextLeft = leftmost
            for child in children:
                nextLeft, pos = _HierarchyPos(
                    G,
                    child,
                    nextLeft,
                    width / len(children),
                    vertGap,
                    vertLoc - vertGap,
                    pos,
                    root,
                )
            pos[root] = (leftmost + width / 2, vertLoc)
            return leftmost + width, pos

    _, pos = _HierarchyPos(G, root, 0, width, vertGap, vertLoc, {})
    pos = AdjustPositions(pos, minHorizontalDistance=1.0, minVerticalDistance=1.0)
    return pos


def AdjustPositions(
    pos, minHorizontalDistance=1.0, minVerticalDistance=1.0, tolerance=1e-4
):
    """
    Adjust node positions to prevent overlapping by ensuring a minimum horizontal and vertical distance between nodes.
    """
    nodes = list(pos.keys())
    adjusted = False
    iteration = 0

    with tqdm(
        total=len(nodes) * (len(nodes) - 1) // 2,
        desc="Adjusting Positions",
        colour="#00274C",
        leave=False,
    ) as pbar:
        while not adjusted:
            adjusted = True
            maxShift = 0
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    node1 = nodes[i]
                    node2 = nodes[j]
                    x1, y1 = pos[node1]
                    x2, y2 = pos[node2]
                    dx = x2 - x1
                    dy = y2 - y1
                    distanceX = abs(dx)
                    distanceY = abs(dy)
                    if (
                        distanceX < minHorizontalDistance
                        and distanceY < minVerticalDistance
                    ):
                        # Shift node2 to the right
                        shiftX = (minHorizontalDistance - distanceX) / 2
                        pos[node2] = (x2 + shiftX, y2)
                        # Shift node1 to the left
                        pos[node1] = (x1 - shiftX, y1)
                        maxShift = max(maxShift, shiftX)
                        adjusted = False
                    pbar.update(1)
            iteration += 1
            if maxShift < tolerance:
                break

    return pos


def VisualizeTraversal(root, traversalFunc, traversalName, speed):
    G = BuildGraph(root)
    pos = HierarchyPos(G, root.value)
    fig, ax = plt.subplots(figsize=(14, 10))
    plt.title(f"{traversalName} Traversal")
    nodeColors = {n: "deepskyblue" for n in G.nodes()}  # Initialize all nodes as blue
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=list(nodeColors.values()),
        node_size=500,
        arrows=True,
        ax=ax,
    )
    plt.axis("off")

    visited = set()
    traversal = list(traversalFunc(root))

    def Update(num):
        event, node = traversal[num]
        if event == "add":
            if node.value not in visited:
                nodeColors[node.value] = "gold"
        elif event == "visit":
            nodeColors[node.value] = "springgreen"
            visited.add(node.value)
        ax.clear()
        plt.title(f"{traversalName} Traversal")
        colors = [nodeColors[n] for n in G.nodes()]
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=colors,
            node_size=500,
            arrows=True,
            ax=ax,
        )
        plt.axis("off")

    ani = animation.FuncAnimation(
        fig, Update, frames=len(traversal), interval=1000 / speed, repeat=False
    )
    ani.save(f"{traversalName}_Traversal.mp4", writer="ffmpeg")
    plt.close()


def BreadthFirstSearch(root):
    queue = deque()
    queue.append(root)
    while queue:
        node = queue.popleft()
        yield ("visit", node)
        for child in node.children:
            yield ("add", child)
            queue.append(child)


def DepthFirstSearch(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield ("visit", node)
        for child in node.children:
            yield ("add", child)
            stack.append(child)


def LevelOrderTraversal(root):
    queue = deque()
    queue.append(root)
    while queue:
        node = queue.popleft()
        yield ("visit", node)
        for child in node.children:
            yield ("add", child)
            queue.append(child)


traversals = [
    (BreadthFirstSearch, "BFS"),
    (DepthFirstSearch, "DFS"),
    (LevelOrderTraversal, "Level Order"),
]


def Main():
    print("=== N-ary Tree Traversal Visualization ===")

    while True:
        try:
            numberOfNodesInput = input(
                "Enter the number of nodes in the tree (positive integer): "
            )
            numberOfNodes = int(numberOfNodesInput)
            if numberOfNodes <= 0:
                print("Number of nodes must be a positive integer. Please try again.")
                continue
            break
        except ValueError:
            print(
                "Invalid input. Please enter a positive integer for the number of nodes."
            )

    while True:
        try:
            maxChildrenInput = input(
                "Enter the maximum number of children per node (positive integer): "
            )
            maxChildren = int(maxChildrenInput)
            if maxChildren < 1 and numberOfNodes > 1:
                print(
                    "Maximum number of children must be at least 1 for trees with more than one node. Please try again."
                )
                continue
            if maxChildren >= numberOfNodes:
                print(
                    f"Maximum number of children per node cannot exceed {numberOfNodes - 1} for a tree with {numberOfNodes} nodes."
                )
                print(
                    "Setting maximum number of children to the maximum feasible value."
                )
                maxChildren = numberOfNodes - 1
                print(f"Maximum number of children set to {maxChildren}")
            break
        except ValueError:
            print(
                "Invalid input. Please enter a positive integer for the maximum number of children."
            )

    while True:
        try:
            speedInput = input(
                "Enter the number of nodes traversed per second (positive number): "
            )
            speed = float(speedInput)
            if speed <= 0:
                print("Speed must be a positive number. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a positive number for the speed.")

    try:
        root = CreateRandomTree(numberOfNodes, maxChildren)
    except ValueError as ve:
        print(f"Error creating tree: {ve}")
        return

    for func, name in tqdm(
        traversals,
        desc="Processing Traversals",
        colour="#00274C",
    ):
        # Removed the filter to visualize all traversals
        VisualizeTraversal(root, func, name, speed)


if __name__ == "__main__":
    Main()
