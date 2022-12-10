// Copyright (c) 2015-2017 Josh Blum
// SPDX-License-Identifier: BSL-1.0

// ** This header should be included first, to avoid compile errors.
// ** At least in the case of the windows header files.

// This header helps to abstract network differences between platforms.
// Including the correct headers for various network APIs.
// And providing various typedefs and definitions when missing.

#pragma once

/***********************************************************************
 * unix socket headers
 **********************************************************************/
#define HAS_UNISTD_H
#ifdef HAS_UNISTD_H
#include <unistd.h> //close
#define closesocket close
#endif //HAS_UNISTD_H

#define HAS_NETDB_H
#ifdef HAS_NETDB_H
#include <netdb.h> //addrinfo
#endif //HAS_NETDB_H

#define HAS_NETINET_IN_H
#ifdef HAS_NETINET_IN_H
#include <netinet/in.h>
#endif //HAS_NETINET_IN_H

#define HAS_NETINET_TCP_H
#ifdef HAS_NETINET_TCP_H
#include <netinet/tcp.h>
#endif //HAS_NETINET_TCP_H

#define HAS_SYS_TYPES_H
#ifdef HAS_SYS_TYPES_H
#include <sys/types.h>
#endif //HAS_SYS_TYPES_H

#define HAS_SYS_SOCKET_H
#ifdef HAS_SYS_SOCKET_H
#include <sys/socket.h>
#endif //HAS_SYS_SOCKET_H

#define HAS_ARPA_INET_H
#ifdef HAS_ARPA_INET_H
#include <arpa/inet.h> //inet_ntop
#endif //HAS_ARPA_INET_H

#define HAS_IFADDRS_H
#ifdef HAS_IFADDRS_H
#include <ifaddrs.h> //getifaddrs
#endif //HAS_IFADDRS_H

#define HAS_NET_IF_H
#ifdef HAS_NET_IF_H
#include <net/if.h> //if_nametoindex
#endif //HAS_NET_IF_H

#define HAS_FCNTL_H
#ifdef HAS_FCNTL_H
#include <fcntl.h> //fcntl and constants
#endif //HAS_FCNTL_H

/***********************************************************************
 * htonll and ntohll for GCC
 **********************************************************************/
#if defined(__GNUC__) && !defined(htonll)
    #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        #define htonll(x) __builtin_bswap64(x)
    #else //big endian
        #define htonll(x) (x)
    #endif //little endian
#endif //__GNUC__ and not htonll

#if defined(__GNUC__) && !defined(ntohll)
    #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        #define ntohll(x) __builtin_bswap64(x)
    #else //big endian
        #define ntohll(x) (x)
    #endif //little endian
#endif //__GNUC__ and not ntohll

/***********************************************************************
 * socket type definitions
 **********************************************************************/
#ifndef INVALID_SOCKET
#define INVALID_SOCKET -1
#endif //INVALID_SOCKET

/***********************************************************************
 * socket errno
 **********************************************************************/
#ifdef _MSC_VER
#define SOCKET_ERRNO WSAGetLastError()
#define SOCKET_EINPROGRESS WSAEWOULDBLOCK
#define SOCKET_ETIMEDOUT WSAETIMEDOUT
#else
#define SOCKET_ERRNO errno
#define SOCKET_EINPROGRESS EINPROGRESS
#define SOCKET_ETIMEDOUT ETIMEDOUT
#endif

/***********************************************************************
 * OSX compatibility
 **********************************************************************/
#if !defined(IPV6_ADD_MEMBERSHIP) && defined(IPV6_JOIN_GROUP)
#define IPV6_ADD_MEMBERSHIP IPV6_JOIN_GROUP
#endif

#if !defined(IPV6_DROP_MEMBERSHIP) && defined(IPV6_LEAVE_GROUP)
#define IPV6_DROP_MEMBERSHIP IPV6_LEAVE_GROUP
#endif

/***********************************************************************
 * socket flag definitions
 **********************************************************************/
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif
