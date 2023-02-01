# write your silhouette score unit tests here
import pytest
from search import graph

def test_bfs_traversal():
    """
    TODO: Write your unit test for a breadth-first
    traversal here. Create an instance of your Graph class 
    using the 'tiny_network.adjlist' file and assert 
    that all nodes are being traversed (ie. returns 
    the right number of nodes, in the right order, etc.)
    """
    tiny_graph = Graph("data/tiny_network.adjlist")
    test_traversal = tiny_graph.bfs("Michael Keiser")
    
    assert test_traversal[-1] == "Charles Chiu", "Last node of Michael Keiser traversal should be Charles Chiu"
    assert test_traversal[-5] == "Neil Risch", "5th to last node of Michael Keiser traversal should be Neil Risch"
    assert len(test_traversal) == 59, "Michael Keiser traversal should have 59 nodes"

def test_bfs():
    """
    TODO: Write your unit test for your breadth-first 
    search here. You should generate an instance of a Graph
    class using the 'citation_network.adjlist' file 
    and assert that nodes that are connected return 
    a (shortest) path between them.
    
    Include an additional test for nodes that are not connected 
    which should return None. 

    """
    cite_graph = Graph("data/citation_network.adjlist")
    empty_graph = Graph("data/empty_network.adjlist")

    test_shortest = cite_graph.bfs("Michael Keiser", "Nevan Krogan")
    true_shortest = ['31422865',
     'Martin Kampmann',
     '29033457',
     'Sourav Bandyopadhyay',
     '29088702',
     'Lani Wu',
     '29657129',
     'Wendell Lim',
     '28481362',
     'Nevan Krogan']

    test_one = cite_graph.bfs("Michael Keiser", "Michael Keiser")
    test_connection = cite_graph.bfs("Michael Keiser", "Bryson Choy")

    try:
        test_in_graph = cite_graph.bfs("Bryson Choy")
    except:
        test_in_graph = False

    try:
        test_no_start = cite_graph.bfs("")
    except:
        test_no_start = True

    try:
        test_empty = empty_graph.bfs("Michael Keiser")
    except:
        test_empty = True



    assert test_shortest == true_shortest, "BFS doesn't find the shortest path between start and end nodes"
    assert test_one == "Michael Keiser", "One-node network should return itself as the shortest path"
    assert test_connection == None, "There shouldn't be any path between these two nodes"
    assert test_in_graph == False, "Start node should be found in the graph"
    assert test_no_start == True, "There should be an error if there is no start node"
    assert test_empty == True, "BFS should not work on an empty graph"
