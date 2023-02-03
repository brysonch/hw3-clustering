# Write your k-means unit tests here
import pytest

def test_fit():
    
    tiny_graph = Graph("data/tiny_network.adjlist")
    test_traversal = tiny_graph.bfs("Michael Keiser")
    
    assert test_traversal[-1] == "Charles Chiu", "Last node of Michael Keiser traversal should be Charles Chiu"
    assert test_traversal[-5] == "Neil Risch", "5th to last node of Michael Keiser traversal should be Neil Risch"
    assert len(test_traversal) == 59, "Michael Keiser traversal should have 59 nodes"

def test_bfs():
    
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
